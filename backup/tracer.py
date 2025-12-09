import json
import os
from pathlib import Path
import sys
import threading
import multiprocessing
import time
import traceback
import concurrent.futures
import multiprocessing.pool
import functools
import signal

import dill
import wcmatch.glob

from types import FrameType, TracebackType
from typing import (
    Callable,
    Any,
    Optional,
    Dict,
    List,
    Type,
    TypedDict,
    Union,
    override,
)

from abc import abstractmethod, ABC

from tracer.utils import (
    deep_equal,
    deep_copy,
    V1RegisterRepresenter,
    no_raise_repr,
    NoRaiseRichRegistryRepresenter,
    SerializableReprWrapperClass,
    HookContext,
)

_StrOrPath = Union[str, Path]


class FunctionDict(TypedDict):
    file: str
    depth: int
    name: str
    params: Optional[Dict[str, Any]]
    globals: Optional[Dict[str, Any]]
    traces: List["FunctionTraceDict"]
    retval: Any
    __local_snapshot__: Optional[Dict[str, Any]]
    __global_snapshot__: Optional[Dict[str, Any]]


class FunctionTraceDict(TypedDict):
    line: int
    locals: Optional[Dict[str, Any]]
    globals: Optional[Dict[str, Any]]
    functions: List["FunctionDict"]
    exceptions: List["FunctionExceptionDict"]


class FunctionExceptionDict(TypedDict):
    function: str
    type: Any
    value: Any
    traceback: Any


class TracerThreadContext:
    def __init__(
        self,
        thread: threading.Thread,
    ):
        self._thread = thread
        self._stacks: List[FunctionDict] = [
            {
                "file": None,
                "depth": None,
                "name": None,
                "params": None,
                "retval": None,
                "traces": [
                    {
                        "line": None,
                        "locals": None,
                        "functions": [],
                        "exceptions": [],
                    }
                ],
            }
        ]
        self._depth = -1


_HOOKS_ = {
    "multiprocessing.Process.__init__": multiprocessing.Process.__init__,
    "multiprocessing.Process.run": multiprocessing.Process.run,
    "concurrent.future.ProcessPoolExecutor.submit": concurrent.futures.ProcessPoolExecutor.submit,
    "concurrent.future.ProcessPoolExecutor.map": concurrent.futures.ProcessPoolExecutor.map,
    "multiprocessing.pool.Pool._map_async": multiprocessing.pool.Pool._map_async,
    "multiprocessing.pool.Pool.apply_async": multiprocessing.pool.Pool.apply_async,
    "multiprocessing.pool.Pool.imap": multiprocessing.pool.Pool.imap,
    "multiprocessing.pool.Pool.imap_unordered": multiprocessing.pool.Pool.imap_unordered,
    "multiprocessing.pool.Pool.__init__": multiprocessing.pool.Pool.__init__,
}


def _tracer_hook_target(
    *args,
    ___tracer_target___,
    ___tracer_cls___,
    ___tracer_kwargs___,
    **kwargs,
):
    tracer = ___tracer_cls___(**___tracer_kwargs___)
    with tracer:
        return ___tracer_target___(*args, **kwargs)


_TRACER_HOOK_WORKER_INSTANCE: Optional["BaseTracer"] = None


def _tracer_hook_worker_stop(
    signum,
    frame,
):
    global _TRACER_HOOK_WORKER_INSTANCE

    if _TRACER_HOOK_WORKER_INSTANCE is not None:
        _TRACER_HOOK_WORKER_INSTANCE.stop()
        _TRACER_HOOK_WORKER_INSTANCE = None

    os._exit(0)


def _tracer_hook_process_worker(
    *args,
    ___tracer_target___,
    ___tracer_cls___,
    ___tracer_kwargs___,
    **kwargs,
):
    tracer = ___tracer_cls___(
        **___tracer_kwargs___,
    )

    def _signal_handler(
        signum,
        frame,
        tracer,
    ):
        tracer.stop()
        os._exit(0)

    signal_handler = functools.partial(
        _signal_handler,
        tracer=tracer,
    )

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    tracer.start()
    try:
        return ___tracer_target___(*args, **kwargs)
    finally:
        tracer.stop()


def _tracer_hook_worker_start(
    ___tracer_target___,
    ___tracer_cls___,
    ___tracer_kwargs___,
    *args,
    **kwargs,
):
    global _TRACER_HOOK_WORKER_INSTANCE

    if _TRACER_HOOK_WORKER_INSTANCE is None:
        _TRACER_HOOK_WORKER_INSTANCE = ___tracer_cls___(**___tracer_kwargs___)
        _TRACER_HOOK_WORKER_INSTANCE.start()

    signal.signal(signal.SIGTERM, _tracer_hook_worker_stop)
    signal.signal(signal.SIGINT, _tracer_hook_worker_stop)

    try:
        return ___tracer_target___(*args, **kwargs)
    finally:
        _tracer_hook_worker_stop(None, None)


class BaseTracer(ABC):
    def __init__(
        self,
        dest_dir: _StrOrPath,
        patterns: Optional[List[_StrOrPath]] = None,
        trace_locals: bool = True,
        trace_globals: bool = False,
        only_trace_changed_locals: bool = True,
        only_trace_changed_globals: bool = True,
        skip_empty_trace: bool = True,
        only_main_thread: bool = False,
        skip_empty_thread: bool = True,
        only_main_process: bool = True,
    ):
        kwargs = locals()
        kwargs.pop("self")
        self._kwargs = kwargs.copy()

        self._is_started = False

        self._dest_dir = Path(dest_dir)
        if self._dest_dir.exists() and not self._dest_dir.is_dir():
            raise ValueError(f"Destination {dest_dir} is not a directory.")

        self._patterns = (
            [wcmatch.glob.compile(p, flags=wcmatch.glob.GLOBSTARLONG) for p in patterns]
            if patterns
            else None
        )

        self._trace_locals = trace_locals
        self._trace_globals = trace_globals
        self._only_trace_changed_locals = only_trace_changed_locals
        self._only_trace_changed_globals = only_trace_changed_globals
        self._skip_empty_trace = skip_empty_trace

        self._only_main_thread = only_main_thread
        self._skip_empty_thread = skip_empty_thread

        self._only_main_process = only_main_process

        self._thread_contexts: Dict[threading.Thread, TracerThreadContext] = {}

        self._hook_context: HookContext = HookContext()

    def __trace__(
        self,
        frame: FrameType,
        event: str,
        arg: Any,
    ) -> Optional[Callable[[FrameType, str, Any], Any]]:

        # update depth
        if event == "call":
            self._set_current_thread_depth(self._get_current_thread_depth() + 1)

        elif event == "return":
            self._set_current_thread_depth(self._get_current_thread_depth() - 1)

        # check patterns
        if self._patterns is not None:
            if not any(
                pattern.match(frame.f_code.co_filename) for pattern in self._patterns
            ):
                return self.__trace__

        if event == "call":

            # create function trace
            function: FunctionDict = {
                "file": frame.f_code.co_filename,
                "depth": self._get_current_thread_depth(),
                "name": frame.f_code.co_name,
                "params": self._serialize_variables(frame.f_locals),
                "traces": [],
                "retval": None,
            }

            # make snapshots
            if self._trace_locals:
                if self._only_trace_changed_locals:
                    function["__local_snapshot__"] = self._copy_variables(
                        frame.f_locals
                    )

            if self._trace_globals:
                function["globals"] = self._serialize_variables(frame.f_globals)
                if self._only_trace_changed_globals:
                    function["__global_snapshot__"] = self._copy_variables(
                        frame.f_globals
                    )

            # push function
            self._current_thread_stack().append(function)

        elif event == "return":

            # pop function
            function = self._current_thread_stack().pop()
            function["retval"] = self._serialize_function_return_value(arg)

            # ensure trace available
            self._ensure_caller_available(
                frame=frame.f_back,
            )

            # remove snapshots
            if self._trace_locals:
                if self._only_trace_changed_locals:
                    del function["__local_snapshot__"]

            if self._trace_globals:
                if self._only_trace_changed_globals:
                    del function["__global_snapshot__"]

            # write function
            self._current_thread_caller()["traces"][-1]["functions"].append(function)

        elif event == "line":

            trace = {
                "line": frame.f_lineno,
                "functions": [],
                "exceptions": [],
            }

            # make trace locals
            if self._trace_locals:
                if self._only_trace_changed_locals:
                    locals = self._diff_variables(
                        self._current_thread_caller()["__local_snapshot__"],
                        frame.f_locals,
                    )
                    self._current_thread_caller()["__local_snapshot__"].update(locals)
                else:
                    locals = frame.f_locals
                locals = self._serialize_variables(locals)
                trace["locals"] = locals

            # make trace globales
            if self._trace_globals:
                if self._only_trace_changed_globals:
                    globals = self._diff_variables(
                        self._current_thread_caller()["__global_snapshot__"],
                        frame.f_globals,
                    )
                    self._current_thread_caller()["__global_snapshot__"].update(globals)
                else:
                    globals = frame.f_globals
                globals = self._serialize_variables(globals)
                trace["globals"] = globals

            # append trace
            if not self._skip_empty_trace or not self._is_trace_empty(trace):
                self._current_thread_caller()["traces"].append(trace)

        elif event == "exception":

            # ensure trace available
            self._ensure_caller_available(
                frame=frame,
            )

            exc_type, exc_value, exc_traceback = arg
            self._current_thread_caller()["traces"][-1]["exceptions"].append(
                {
                    "function": frame.f_code.co_name,
                    "type": self._serialize_exception_type(exc_type),
                    "value": self._serialize_exception_value(exc_value),
                    "traceback": self._serialize_exception_traceback(exc_traceback),
                }
            )

        return self.__trace__

    def _current_thread_context(
        self,
    ) -> TracerThreadContext:
        thread = threading.current_thread()
        if thread not in self._thread_contexts:
            self._thread_contexts[thread] = TracerThreadContext(thread)
        return self._thread_contexts[thread]

    def _get_current_thread_depth(
        self,
    ) -> int:
        return self._current_thread_context()._depth

    def _set_current_thread_depth(
        self,
        depth: int,
    ) -> None:
        self._current_thread_context()._depth = depth

    def _current_thread_stack(
        self,
    ) -> List[FunctionDict]:
        return self._current_thread_context()._stacks

    def _current_thread_caller(
        self,
    ) -> FunctionDict:
        return self._current_thread_stack()[-1]

    def _is_trace_empty(
        self,
        trace: FunctionTraceDict,
    ) -> bool:
        for key in [
            "locals",
            "globals",
            "functions",
            "exceptions",
        ]:
            if key in trace and trace[key]:
                return False
        return True

    def _ensure_caller_available(
        self,
        frame: FrameType,
    ) -> None:

        if self._current_thread_caller()["traces"]:
            return

        trace = {
            "line": frame.f_lineno,
            "functions": [],
            "exceptions": [],
        }

        if self._trace_locals:
            if self._only_trace_changed_locals:
                locals = self._diff_variables(
                    self._current_thread_caller()["__local_snapshot__"], frame.f_locals
                )
                self._current_thread_caller()["__local_snapshot__"].update(locals)
            else:
                locals = frame.f_locals
            locals = self._serialize_variables(locals)
            trace["locals"] = locals

        if self._trace_globals:
            if self._only_trace_changed_globals:
                globals = self._diff_variables(
                    self._current_thread_caller()["__global_snapshot__"],
                    frame.f_globals,
                )
                self._current_thread_caller()["__global_snapshot__"].update(globals)
            else:
                globals = frame.f_globals
            globals = self._serialize_variables(globals)
            trace["globals"] = globals

        self._current_thread_caller()["traces"].append(trace)

    def _start_trace(
        self,
    ) -> None:
        if self._only_main_thread:
            sys.settrace(self.__trace__)
        else:
            threading.settrace_all_threads(self.__trace__)

    def _stop_trace(
        self,
    ) -> None:
        if self._only_main_thread:
            sys.settrace(None)
        else:
            threading.settrace_all_threads(None)

    def _start_hook(
        self,
    ) -> None:

        if self._only_main_process:
            return

        # def _multiprocessing_process_init_(
        #     p_self,
        #     group=None,
        #     target=None,
        #     name=None,
        #     args=(),
        #     kwargs={},
        #     *,
        #     daemon=None,
        # ):

        #     if target is None and p_self.run != _HOOKS_["multiprocessing.Process.run"]:
        #         target = p_self.run
        #         args = ()
        #         kwargs = {}
        #         p_self.run = _HOOKS_["multiprocessing.Process.run"]

        #     elif target is None:
        #         return _HOOKS_["multiprocessing.Process.__init__"](
        #             p_self,
        #             group=group,
        #             target=target,
        #             name=name,
        #             args=args,
        #             kwargs=kwargs,
        #             daemon=daemon,
        #         )

        #     args = (
        #         target,
        #         self.__class__,
        #         self._kwargs,
        #     ) + args

        #     return _HOOKS_["multiprocessing.Process.__init__"](
        #         p_self,
        #         group=group,
        #         target=_tracer_hook_worker_start,
        #         name=name,
        #         args=args,
        #         kwargs=kwargs,
        #         daemon=daemon,
        #     )

        # def _concurrent_future_processpoolexecutor_submit_(
        #     p_self,
        #     fn: Callable,
        #     *args,
        #     **kwargs,
        # ) -> concurrent.futures.Future:

        #     wrapped_func = functools.partial(
        #         _tracer_hook_target,
        #         ___tracer_target___=fn,
        #         ___tracer_cls___=self.__class__,
        #         ___tracer_kwargs___=self._kwargs,
        #     )

        #     return _HOOKS_["concurrent.future.ProcessPoolExecutor.submit"](
        #         p_self,
        #         wrapped_func,
        #         *args,
        #         **kwargs,
        #     )

        # def _concurrent_future_processpoolexecutor_map_(
        #     p_self,
        #     fn: Callable,
        #     *iterables,
        #     timeout=None,
        #     chunksize=1,
        # ):
        #     wrapped_func = functools.partial(
        #         _tracer_hook_target,
        #         ___tracer_target___=fn,
        #         ___tracer_cls___=self.__class__,
        #         ___tracer_kwargs___=self._kwargs,
        #     )

        #     return _HOOKS_["concurrent.future.ProcessPoolExecutor.map"](
        #         p_self,
        #         wrapped_func,
        #         *iterables,
        #         timeout=timeout,
        #         chunksize=chunksize,
        #     )

        # def _multiprocessing_pool_pool_map_async_(
        #     p_self,
        #     func: Callable,
        #     iterable,
        #     mapper,
        #     chunksize=None,
        #     callback=None,
        #     error_callback=None,
        # ):
        #     wrapped_func = functools.partial(
        #         _tracer_hook_target,
        #         ___tracer_target___=func,
        #         ___tracer_cls___=self.__class__,
        #         ___tracer_kwargs___=self._kwargs,
        #     )

        #     return _HOOKS_["multiprocessing.pool.Pool._map_async"](
        #         p_self,
        #         wrapped_func,
        #         iterable,
        #         mapper,
        #         chunksize=chunksize,
        #         callback=callback,
        #         error_callback=error_callback,
        #     )

        # def _multiprocessing_pool_pool_apply_async_(
        #     p_self,
        #     func,
        #     args=(),
        #     kwds={},
        #     callback=None,
        #     error_callback=None,
        # ):
        #     wrapped_func = functools.partial(
        #         _tracer_hook_target,
        #         ___tracer_target___=func,
        #         ___tracer_cls___=self.__class__,
        #         ___tracer_kwargs___=self._kwargs,
        #     )

        #     return _HOOKS_["multiprocessing.pool.Pool.apply_async"](
        #         p_self,
        #         wrapped_func,
        #         args=args,
        #         kwds=kwds,
        #         callback=callback,
        #         error_callback=error_callback,
        #     )

        # def _multiprocessing_pool_pool_imap_(
        #     p_self,
        #     func,
        #     iterable,
        #     chunksize=1,
        # ):
        #     wrapped_func = functools.partial(
        #         _tracer_hook_target,
        #         ___tracer_target___=func,
        #         ___tracer_cls___=self.__class__,
        #         ___tracer_kwargs___=self._kwargs,
        #     )

        #     return _HOOKS_["multiprocessing.pool.Pool.imap"](
        #         p_self,
        #         wrapped_func,
        #         iterable,
        #         chunksize=chunksize,
        #     )

        # def _multiprocessing_pool_pool_imap_unordered_(
        #     p_self,
        #     func,
        #     iterable,
        #     chunksize=1,
        # ):
        #     wrapped_func = functools.partial(
        #         _tracer_hook_target,
        #         ___tracer_target___=func,
        #         ___tracer_cls___=self.__class__,
        #         ___tracer_kwargs___=self._kwargs,
        #     )

        #     return _HOOKS_["multiprocessing.pool.Pool.imap_unordered"](
        #         p_self,
        #         wrapped_func,
        #         iterable,
        #         chunksize=chunksize,
        #     )

        def _multiprocess_process_init_before(
            kwargs: Dict,
            context: Dict,
        ):
            target: Callable = kwargs.get("target", None)

            def _target(
                *args,
                ___tracer_target___,
                ___tracer_cls___,
                ___tracer_kwargs___,
                **kwargs,
            ):
                tracer = ___tracer_cls___(
                    **___tracer_kwargs___,
                )

                def _signal_handler(
                    signum,
                    frame,
                    tracer,
                ):
                    tracer.stop()
                    os._exit(0)

                signal_handler = functools.partial(
                    _signal_handler,
                    tracer=tracer,
                )

                signal.signal(signal.SIGTERM, signal_handler)
                signal.signal(signal.SIGINT, signal_handler)

                tracer.start()
                try:
                    return ___tracer_target___(*args, **kwargs)
                finally:
                    tracer.stop()

            target = functools.partial(
                _target,
                ___tracer_target___=target,
                ___tracer_cls___=self.__class__,
                ___tracer_kwargs___=self._kwargs,
            )

            return kwargs

        # self._hook_context.hook(
        #     obj=multiprocessing.Process,
        #     name="__init__",
        #     new_func=_multiprocessing_process_init_,
        # )

        # setattr(
        #     multiprocessing.Process,
        #     "__init__",
        #     _multiprocessing_process_init_,
        # )
        # setattr(
        #     concurrent.futures.ProcessPoolExecutor,
        #     "submit",
        #     _concurrent_future_processpoolexecutor_submit_,
        # )
        # setattr(
        #     concurrent.futures.ProcessPoolExecutor,
        #     "map",
        #     _concurrent_future_processpoolexecutor_map_,
        # )
        # setattr(
        #     multiprocessing.pool.Pool,
        #     "_map_async",
        #     _multiprocessing_pool_pool_map_async_,
        # )
        # setattr(
        #     multiprocessing.pool.Pool,
        #     "apply_async",
        #     _multiprocessing_pool_pool_apply_async_,
        # )
        # setattr(
        #     multiprocessing.pool.Pool,
        #     "imap",
        #     _multiprocessing_pool_pool_imap_,
        # )
        # setattr(
        #     multiprocessing.pool.Pool,
        #     "imap_unordered",
        #     _multiprocessing_pool_pool_imap_unordered_,
        # )

    def _stop_hook(
        self,
    ) -> None:

        if self._only_main_process:
            return

        self._hook_context.unhook_all()

        # setattr(
        #     multiprocessing.Process,
        #     "__init__",
        #     _HOOKS_["multiprocessing.Process.__init__"],
        # )
        # setattr(
        #     concurrent.futures.ProcessPoolExecutor,
        #     "submit",
        #     _HOOKS_["concurrent.future.ProcessPoolExecutor.submit"],
        # )
        # setattr(
        #     concurrent.futures.ProcessPoolExecutor,
        #     "map",
        #     _HOOKS_["concurrent.future.ProcessPoolExecutor.map"],
        # )
        # setattr(
        #     multiprocessing.pool.Pool,
        #     "_map_async",
        #     _HOOKS_["multiprocessing.pool.Pool._map_async"],
        # )
        # setattr(
        #     multiprocessing.pool.Pool,
        #     "apply_async",
        #     _HOOKS_["multiprocessing.pool.Pool.apply_async"],
        # )
        # setattr(
        #     multiprocessing.pool.Pool,
        #     "imap",
        #     _HOOKS_["multiprocessing.pool.Pool.imap"],
        # )
        # setattr(
        #     multiprocessing.pool.Pool,
        #     "imap_unordered",
        #     _HOOKS_["multiprocessing.pool.Pool.imap_unordered"],
        # )

    def _stop_save(
        self,
    ) -> None:
        if self._dest_dir:
            self._dest_dir.mkdir(parents=True, exist_ok=True)
            for thread, trace in self._get_results().items():
                with open(
                    self._dest_dir
                    / f"tracer__ppid-{os.getppid()}__pid-{os.getpid()}__thread-{thread.name}__{time.time()}.json",
                    "w",
                ) as f:
                    json.dump(trace, f, indent=4)

    def start(self):
        if self._is_started:
            return
        self._is_started = True
        self._start_trace()
        self._start_hook()

    def stop(self):
        if not self._is_started:
            return
        self._is_started = False
        self._stop_trace()
        self._stop_hook()
        self._stop_save()

    def __enter__(self):
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException],
        exc_value: BaseException,
        traceback: TracebackType,
    ):
        self.stop()

    def _get_results(
        self,
    ) -> Dict[threading.Thread, Dict]:
        results = {}
        for thread, value in self._thread_contexts.items():

            # repair broken stacks due to abnormal termination
            while len(value._stacks) > 1:
                function = value._stacks.pop()
                function["retval"] = "<function is terminated>"

                if self._trace_locals:
                    if self._only_trace_changed_locals:
                        del function["__local_snapshot__"]

                if self._trace_globals:
                    if self._only_trace_changed_globals:
                        del function["__global_snapshot__"]

                value._stacks[-1]["traces"][-1]["functions"].append(function)

            value = value._stacks[0]["traces"][0]["functions"]
            if self._skip_empty_thread and not value:
                continue
            results[thread] = value
        return results

    @abstractmethod
    def _serialize_variable(
        self,
        variable: Any,
    ) -> str: ...

    def _serialize_variables(
        self,
        variables: Dict[str, Any],
    ) -> Dict[str, str]:
        return {k: self._serialize_variable(v) for k, v in variables.items()}

    @abstractmethod
    def _serialize_function_return_value(
        self,
        function_return_value: Any,
    ) -> str: ...

    @abstractmethod
    def _serialize_exception_type(
        self,
        exception_type: Type[BaseException],
    ) -> str: ...

    @abstractmethod
    def _serialize_exception_value(
        self,
        exception_value: BaseException,
    ) -> str: ...

    @abstractmethod
    def _serialize_exception_traceback(
        self,
        exception_traceback: TracebackType,
    ) -> str: ...

    @abstractmethod
    def _compare_variable(
        self,
        v1: Any,
        v2: Any,
    ) -> bool: ...

    @abstractmethod
    def _copy_variable(
        self,
        variable: Any,
    ) -> Any: ...

    def _copy_variables(
        self,
        variables: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {k: self._copy_variable(v) for k, v in variables.items()}

    @abstractmethod
    def _diff_variables(
        self,
        snapshot: Dict[str, Any],
        variables: Dict[str, Any],
    ) -> Dict[str, Any]: ...


class _V1ReprTracer(BaseTracer):

    @override
    def _serialize_variable(
        self,
        variable: Any,
    ) -> Any:
        return no_raise_repr(variable)

    @override
    def _serialize_function_return_value(
        self,
        function_return_value: Any,
    ) -> str:
        return no_raise_repr(function_return_value)

    @override
    def _serialize_exception_type(
        self,
        exception_type: Type[BaseException],
    ) -> Any:
        return no_raise_repr(exception_type)

    @override
    def _serialize_exception_value(
        self,
        exception_value: BaseException,
    ) -> Any:
        return no_raise_repr(exception_value)

    @override
    def _serialize_exception_traceback(
        self,
        exception_traceback: TracebackType,
    ) -> Any:
        return "\n".join(traceback.format_tb(exception_traceback))

    @override
    def _compare_variable(
        self,
        v1: Any,
        v2: Any,
    ) -> bool:
        return deep_equal(v1, v2)

    @override
    def _copy_variable(
        self,
        variable: Any,
    ) -> Any:
        try:
            return deep_copy(variable)
        except:
            return no_raise_repr(variable)

    @override
    def _diff_variables(
        self,
        snapshot: Dict[str, Any],
        variables: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self._copy_variables(
            {
                k: v
                for k, v in variables.items()
                if k not in snapshot or not self._compare_variable(snapshot[k], v)
            }
        )


class _V2ReprTracer(BaseTracer):
    def __init__(
        self,
        dest_dir: _StrOrPath,
        patterns: Optional[List[_StrOrPath]] = None,
        trace_locals: bool = True,
        trace_globals: bool = False,
        only_trace_changed_locals: bool = True,
        only_trace_changed_globals: bool = True,
        skip_empty_trace: bool = True,
        only_main_thread: bool = False,
        skip_empty_thread: bool = True,
        only_main_process: bool = False,
    ):
        super().__init__(
            dest_dir=dest_dir,
            patterns=patterns,
            trace_locals=trace_locals,
            trace_globals=trace_globals,
            only_trace_changed_locals=only_trace_changed_locals,
            only_trace_changed_globals=only_trace_changed_globals,
            skip_empty_trace=skip_empty_trace,
            only_main_thread=only_main_thread,
            skip_empty_thread=skip_empty_thread,
            only_main_process=only_main_process,
        )

        self._helper = NoRaiseRichRegistryRepresenter()

    def helper(
        self,
    ) -> V1RegisterRepresenter:
        return self._helper

    @override
    def _serialize_variable(
        self,
        variable: Any,
    ) -> Any:
        return self.repr(variable)

    @override
    def _serialize_function_return_value(
        self,
        function_return_value: Any,
    ) -> str:
        return self.repr(function_return_value)

    @override
    def _serialize_exception_type(
        self,
        exception_type: Type[BaseException],
    ) -> Any:
        return self.repr(exception_type)

    @override
    def _serialize_exception_value(
        self,
        exception_value: BaseException,
    ) -> Any:
        return self.repr(exception_value)

    @override
    def _serialize_exception_traceback(
        self,
        exception_traceback: TracebackType,
    ) -> Any:
        return "\n".join(traceback.format_tb(exception_traceback))

    @override
    def _compare_variable(
        self,
        v1: Any,
        v2: Any,
    ) -> bool:
        return deep_equal(v1, v2)

    @override
    def _copy_variable(
        self,
        variable: Any,
    ) -> Any:
        try:
            return deep_copy(variable)
        except:
            cls = type(variable)
            try:
                cls_name = cls.__name__
            except:
                cls_name = "[unknown_class_name]"
            try:
                cls_module = cls.__module__
            except:
                cls_module = "[unknown_class_module]"
            return f"<uncopyable {cls_module}.{cls_name} object>"

    @override
    def _diff_variables(
        self,
        snapshot: Dict[str, Any],
        variables: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self._copy_variables(
            {
                k: v
                for k, v in variables.items()
                if k not in snapshot or not self._compare_variable(snapshot[k], v)
            }
        )


class _V3ReprTracer(BaseTracer):

    def __init__(
        self,
        dest_dir: _StrOrPath,
        patterns: Optional[List[_StrOrPath]] = None,
        trace_locals: bool = True,
        trace_globals: bool = False,
        only_trace_changed_locals: bool = True,
        only_trace_changed_globals: bool = True,
        skip_empty_trace: bool = True,
        only_main_thread: bool = False,
        skip_empty_thread: bool = True,
        only_main_process: bool = False,
    ):
        super().__init__(
            dest_dir=dest_dir,
            patterns=patterns,
            trace_locals=trace_locals,
            trace_globals=trace_globals,
            only_trace_changed_locals=only_trace_changed_locals,
            only_trace_changed_globals=only_trace_changed_globals,
            skip_empty_trace=skip_empty_trace,
            only_main_thread=only_main_thread,
            skip_empty_thread=skip_empty_thread,
            only_main_process=only_main_process,
        )

        self._helper = NoRaiseRichRegistryRepresenter()

    def repr(
        self,
        var: Any,
    ) -> str:
        return self._helper.repr(var)

    def helper(
        self,
    ) -> V1RegisterRepresenter:
        return self._helper

    @override
    def _serialize_variable(
        self,
        variable: Any,
    ) -> Any:
        return self.repr(variable)

    @override
    def _serialize_function_return_value(
        self,
        function_return_value: Any,
    ) -> str:
        return self.repr(function_return_value)

    @override
    def _serialize_exception_type(
        self,
        exception_type: Type[BaseException],
    ) -> Any:
        return self.repr(exception_type)

    @override
    def _serialize_exception_value(
        self,
        exception_value: BaseException,
    ) -> Any:
        return self.repr(exception_value)

    @override
    def _serialize_exception_traceback(
        self,
        exception_traceback: TracebackType,
    ) -> Any:
        return "\n".join(traceback.format_tb(exception_traceback))

    @override
    def _compare_variable(
        self,
        v1: Any,
        v2: Any,
    ) -> bool:
        try:
            return dill.dumps(v1) == dill.dumps(v2)
        except:
            return self.repr(v1) == self.repr(v2)

    @override
    def _copy_variable(
        self,
        variable: Any,
    ) -> Any:
        try:
            return dill.loads(dill.dumps(variable))
        except:
            return SerializableReprWrapperClass(variable)

    @override
    def _diff_variables(
        self,
        snapshot: Dict[str, Any],
        variables: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self._copy_variables(
            {
                k: v
                for k, v in variables.items()
                if k not in snapshot or not self._compare_variable(snapshot[k], v)
            }
        )


Tracer = _V3ReprTracer
