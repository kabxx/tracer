import json
import os
from pathlib import Path
import sys
import threading
import traceback
import functools
import signal
import threading


from types import FrameType, TracebackType
from typing import (
    Callable,
    Any,
    Literal,
    Optional,
    Dict,
    List,
    Type,
    TypedDict,
    Union,
)

from typing_extensions import override

from enum import Enum
from abc import abstractmethod, ABC

from tracer.utils import (
    HookContext,
    Serializer,
    Representer,
    safe_repr,
    get_type_full_name,
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
        self._thread_name = thread.name
        self._thread_id = thread.ident
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


class BaseTracer(ABC):

    class Status(Enum):
        UNKNOWN = -1
        INIT = 0
        STARTED = 1
        STOPPED = 2

    def __init__(
        self,
        dest_dir: _StrOrPath,
        includes: Optional[List[_StrOrPath]] = None,
        excludes: Optional[List[_StrOrPath]] = None,
        trace_locals: bool = True,
        trace_globals: bool = False,
        only_trace_changed_locals: bool = True,
        only_trace_changed_globals: bool = True,
        skip_empty_trace: bool = True,
        only_main_thread: bool = False,
        skip_empty_thread: bool = True,
        only_main_process: bool = False,
    ):
        kwds = locals()
        kwds.pop("self")
        self._kwds = kwds.copy()

        self._status: BaseTracer.Status = BaseTracer.Status.INIT

        self._dest_dir = Path(dest_dir)
        if self._dest_dir.exists() and not self._dest_dir.is_dir():
            raise ValueError(f"Destination {dest_dir} is not a directory.")

        self._includes = [str(p) for p in includes] if includes else None
        self._excludes = [str(p) for p in excludes] if excludes else None

        self._trace_locals = trace_locals
        self._trace_globals = trace_globals
        self._only_trace_changed_locals = only_trace_changed_locals
        self._only_trace_changed_globals = only_trace_changed_globals
        self._skip_empty_trace = skip_empty_trace

        # trace thread
        self._only_main_thread = only_main_thread
        self._skip_empty_thread = skip_empty_thread
        # self._thread_contexts: Dict[threading.Thread, TracerThreadContext] = {}

        self._thread_lock = threading.Lock()
        self._thread_local = threading.local()
        self._thread_contexts: List[TracerThreadContext] = []

        # trace process
        self._only_main_process = only_main_process

        # hook
        self._hook_context: HookContext = HookContext()

    def trace(
        self,
        frame: FrameType,
        event: str,
        arg: Any,
    ) -> Optional[Callable[[FrameType, str, Any], Any]]:

        # print(".",flush=False,end="")

        # update depth
        if event == "call":
            self._current_thread_depth_set(self._current_thread_depth_get() + 1)

        elif event == "return":
            self._current_thread_depth_set(self._current_thread_depth_get() - 1)

        # check patterns
        if self._includes is not None:
            if not any(
                frame.f_code.co_filename.startswith(pattern)
                for pattern in self._includes
            ):
                return self.trace
        if self._excludes is not None:
            if any(
                frame.f_code.co_filename.startswith(pattern)
                for pattern in self._excludes
            ):
                return self.trace

        # if event == "call" and True:
        #     print(
        #         f"TRACE CALL [{self._current_thread_depth_get()}] {frame.f_lineno} {frame.f_code.co_name} IN {frame.f_code.co_filename} TIME {time.time()}",
        #         flush=False,
        #     )  # DEBUG

        if event == "call":
            locals = self._snapshot_variables(frame.f_locals)

            # create function trace
            function: FunctionDict = {
                "file": frame.f_code.co_filename,
                "depth": self._current_thread_depth_get(),
                "name": frame.f_code.co_name,
                "params": self._output_from_snapshot_variables(locals),
                "traces": [],
                "retval": None,
            }

            # make snapshots
            if self._trace_locals:
                if self._only_trace_changed_locals:
                    function["__local_snapshot__"] = locals
            if self._trace_globals:
                globals = self._snapshot_variables(frame.f_globals)
                if self._only_trace_changed_globals:
                    function["__global_snapshot__"] = globals
                function["globals"] = self._output_from_snapshot_variables(globals)

            # push function
            self._current_thread_stack().append(function)

        elif event == "return":

            # pop function
            function = self._current_thread_stack().pop()
            function["retval"] = self._output_from_snapshot_variable(
                self._snapshot_variable(arg)
            )

            # ensure trace available
            self._current_thread_ensure_caller_legal(
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
                locals = self._snapshot_variables(frame.f_locals)
                if self._only_trace_changed_locals:
                    locals = self._diff_snapshot_variables(
                        self._current_thread_caller()["__local_snapshot__"],
                        locals,
                    )
                    self._current_thread_caller()["__local_snapshot__"].update(locals)
                trace["locals"] = self._output_from_snapshot_variables(locals)

            # make trace globales
            if self._trace_globals:
                globals = self._snapshot_variables(frame.f_globals)
                if self._only_trace_changed_globals:
                    globals = self._diff_snapshot_variables(
                        self._current_thread_caller()["__global_snapshot__"],
                        globals,
                    )
                    self._current_thread_caller()["__global_snapshot__"].update(globals)
                trace["globals"] = self._output_from_snapshot_variables(globals)

            # append trace
            if not self._skip_empty_trace or not self._is_function_trace_empty(trace):
                self._current_thread_caller()["traces"].append(trace)

        elif event == "exception":
            # ensure trace available
            self._current_thread_ensure_caller_legal(
                frame=frame,
            )

            exc_type, exc_value, exc_traceback = arg
            self._current_thread_caller()["traces"][-1]["exceptions"].append(
                {
                    "function": frame.f_code.co_name,
                    "type": self._output_by_exc_type(exc_type),
                    "value": self._output_by_exc_val(exc_value),
                    "traceback": self._output_by_exc_traceback(exc_traceback),
                }
            )

        return self.trace

    def _current_thread_context(
        self,
    ) -> TracerThreadContext:
        if hasattr(self._thread_local, "context"):
            return getattr(self._thread_local, "context")
        thread_context = TracerThreadContext(thread=threading.current_thread())
        setattr(self._thread_local, "context", thread_context)
        with self._thread_lock:
            self._thread_contexts.append(thread_context)
        return thread_context

    def _current_thread_depth_get(
        self,
    ) -> int:
        return self._current_thread_context()._depth

    def _current_thread_depth_set(
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

    def _current_thread_ensure_caller_legal(
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
            locals = self._snapshot_variables(frame.f_locals)
            if self._only_trace_changed_locals:
                locals = self._diff_snapshot_variables(
                    self._current_thread_caller()["__local_snapshot__"], locals
                )
                self._current_thread_caller()["__local_snapshot__"].update(locals)
            trace["locals"] = self._output_from_snapshot_variables(locals)

        if self._trace_globals:
            globals = self._snapshot_variables(frame.f_globals)
            if self._only_trace_changed_globals:
                globals = self._diff_snapshot_variables(
                    self._current_thread_caller()["__global_snapshot__"],
                    globals,
                )
                self._current_thread_caller()["__global_snapshot__"].update(globals)
            trace["globals"] = self._output_from_snapshot_variables(globals)

        self._current_thread_caller()["traces"].append(trace)

    def _is_function_trace_empty(
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

    def _start_trace(
        self,
    ) -> None:
        if self._only_main_thread:
            sys.settrace(self.trace)
        else:
            threading.settrace_all_threads(self.trace)

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

        def _hook(
            *args,
            ___tracer_target___,
            ___tracer_cls___,
            ___tracer_kwds___,
            **kwds,
        ):
            print(f"PROCESS {os.getpid()} START")

            tracer = ___tracer_cls___(
                **___tracer_kwds___,
            )

            def _signal_handler(
                signum,
                frame,
                tracer,
            ):
                tracer.stop()
                print(f"PROCESS {os.getpid()} EXIT ON SIGNAL {signum}")
                os._exit(0)

            signal_handler = functools.partial(
                _signal_handler,
                tracer=tracer,
            )

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            tracer.start()
            try:
                return ___tracer_target___(*args, **kwds)
            finally:
                print(f"PROCESS {os.getpid()} STOP")
                tracer.stop()

        def _multiprocessing_process_init_before(
            _: Dict,
            kwds: Dict,
        ):
            target = kwds.get("target", None)

            if target is None:
                process = kwds.get("self", None)
                target = getattr(process, "run")
                setattr(
                    process,
                    "run",
                    functools.partial(
                        getattr(multiprocessing.process.BaseProcess, "run"),
                        process,
                    ),
                )

            kwds["target"] = functools.partial(
                _hook,
                ___tracer_target___=target,
                ___tracer_cls___=self.__class__,
                ___tracer_kwds___=self._kwds,
            )

        try:
            import multiprocessing.process

            self._hook_context.hook(
                obj=multiprocessing.process.BaseProcess,
                name="__init__",
                before=_multiprocessing_process_init_before,
            )
        except ImportError:
            pass

        try:
            import multiprocess.process

            self._hook_context.hook(
                obj=multiprocess.process.BaseProcess,
                name="__init__",
                before=_multiprocessing_process_init_before,
            )
        except ImportError:
            pass

    def _stop_hook(
        self,
    ) -> None:

        if self._only_main_process:
            return

        self._hook_context.unhook_all()

    def _stop_save(
        self,
    ) -> None:
        if self._dest_dir:
            self._dest_dir.mkdir(parents=True, exist_ok=True)
            for (thread_id, thread_name), trace in self._make_results().items():
                with open(
                    self._dest_dir
                    / f"tracer__ppid-{os.getppid()}__pid-{os.getpid()}__thread-{thread_name}.json",
                    "w",
                ) as f:
                    json.dump(trace, f, indent=4)

    def start(
        self,
    ):
        if self._status == BaseTracer.Status.STOPPED:
            raise RuntimeError("Tracer has been stopped and cannot be restarted.")
        if self._status == BaseTracer.Status.STARTED:
            return
        self._status = BaseTracer.Status.UNKNOWN
        self._start_trace()
        self._start_hook()
        self._status = BaseTracer.Status.STARTED

    def stop(
        self,
    ):
        if self._status == BaseTracer.Status.INIT:
            raise RuntimeError("Tracer has not been started.")
        if self._status == BaseTracer.Status.STOPPED:
            return
        self._status = BaseTracer.Status.UNKNOWN
        self._stop_trace()
        self._stop_hook()
        self._stop_save()
        self._status = BaseTracer.Status.STOPPED

    def __enter__(
        self,
    ):
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException],
        exc_value: BaseException,
        traceback: TracebackType,
    ):
        self.stop()

    def _make_results(
        self,
    ) -> Dict[threading.Thread, Dict]:
        results = {}

        for thread_context in self._thread_contexts:

            # repair broken stacks due to abnormal termination
            while len(thread_context._stacks) > 1:
                function = thread_context._stacks.pop()
                function["retval"] = "<function is terminated>"

                if self._trace_locals:
                    if self._only_trace_changed_locals:
                        del function["__local_snapshot__"]

                if self._trace_globals:
                    if self._only_trace_changed_globals:
                        del function["__global_snapshot__"]

                if not thread_context._stacks[-1]["traces"]:
                    thread_context._stacks[-1]["traces"].append(
                        {
                            "line": "<unknown line>",
                            "locals": "<unknown locals>",
                            "functions": [],
                            "exceptions": [],
                        }
                    )

                thread_context._stacks[-1]["traces"][-1]["functions"].append(function)
            # repair broken stacks end

            value = thread_context._stacks[0]["traces"][0]["functions"]

            if self._skip_empty_thread and not value:
                continue

            results[(thread_context._thread_id, thread_context._thread_name)] = value

        return results

    @abstractmethod
    def _output_from_snapshot_variable(
        self,
        variable: Any,
    ) -> str: ...

    def _output_from_snapshot_variables(
        self,
        variables: Dict[str, Any],
    ) -> Dict[str, str]:
        return {k: self._output_from_snapshot_variable(v) for k, v in variables.items()}

    @abstractmethod
    def _output_by_exc_type(
        self,
        exception_type: Type[BaseException],
    ) -> str: ...

    @abstractmethod
    def _output_by_exc_val(
        self,
        exception_value: BaseException,
    ) -> str: ...

    def _output_by_exc_traceback(
        self,
        exception_traceback: TracebackType,
    ) -> str:
        return "\n".join(traceback.format_tb(exception_traceback))

    @abstractmethod
    def _snapshot_variable(
        self,
        variable: Any,
    ) -> Any: ...

    def _snapshot_variables(
        self,
        variables: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {k: self._snapshot_variable(v) for k, v in variables.items()}

    @abstractmethod
    def _compare_snapshot_variable(
        self,
        v1: Any,
        v2: Any,
    ) -> bool: ...

    @abstractmethod
    def _diff_snapshot_variables(
        self,
        snapshot: Dict[str, Any],
        variables: Dict[str, Any],
    ) -> Dict[str, Any]: ...


class ReprTracer(BaseTracer):

    def __init__(
        self,
        dest_dir: _StrOrPath,
        includes: Optional[List[_StrOrPath]] = None,
        excludes: Optional[List[_StrOrPath]] = None,
        trace_locals: bool = True,
        trace_globals: bool = False,
        only_trace_changed_locals: bool = True,
        only_trace_changed_globals: bool = True,
        skip_empty_trace: bool = True,
        only_main_thread: bool = False,
        skip_empty_thread: bool = True,
        only_main_process: bool = False,
        serializer_backend: Literal["pickle", "dill"] = "pickle",
        skip_types: Optional[List[str]] = None,
    ):
        super().__init__(
            dest_dir=dest_dir,
            includes=includes,
            excludes=excludes,
            trace_locals=trace_locals,
            trace_globals=trace_globals,
            only_trace_changed_locals=only_trace_changed_locals,
            only_trace_changed_globals=only_trace_changed_globals,
            skip_empty_trace=skip_empty_trace,
            only_main_thread=only_main_thread,
            skip_empty_thread=skip_empty_thread,
            only_main_process=only_main_process,
        )

        self._skip_types = skip_types or []

        self._representer = Representer()

        self._serializer = Serializer(
            backend=serializer_backend,
        )

    class SnapShotVariable:
        def __init__(
            self,
            type: Any,
            repr: str,
            bytes: bytes,
        ):
            self._type = type
            self._repr = repr
            self._bytes = bytes

        def __bytes__(
            self,
        ):
            return self._bytes

        def __repr__(
            self,
        ):
            return self._repr

        def __eq__(
            self,
            value: "ReprTracer.SnapShotVariable",
        ):
            return self._type == value._type and bytes(self) == bytes(value)

    def serializer(
        self,
    ) -> Serializer:
        return self._serializer

    def representer(
        self,
    ) -> Representer:
        return self._representer

    def repr(
        self,
        var: Any,
    ) -> str:
        return self.representer().repr(var)

    def serialize(
        self,
        var: Any,
    ) -> bytes:
        return self.serializer().dumps(var)

    def deserialize(
        self,
        data: bytes,
    ) -> Any:
        return self.serializer().loads(data)

    @override
    def _output_by_exc_type(
        self,
        exception_type: Type[BaseException],
    ) -> str:
        return self.repr(exception_type)

    @override
    def _output_by_exc_val(
        self,
        exception_value: BaseException,
    ) -> str:
        return self.repr(exception_value)

    @override
    def _snapshot_variable(
        self,
        variable: Any,
    ) -> SnapShotVariable:

        cls = type(variable)
        cls_full_name = get_type_full_name(cls)
        # skip
        if not cls_full_name or any(
            cls_full_name.startswith(prefix) for prefix in self._skip_types
        ):
            repr = safe_repr(variable)
            bytes = self.serialize(repr)
            return self.SnapShotVariable(cls, repr, bytes)
        # simple
        repr = self.repr(variable)
        try:
            bytes = self.serialize(variable)
        except:
            bytes = self.serialize(repr)
        return self.SnapShotVariable(cls, repr, bytes)

    @override
    def _output_from_snapshot_variable(
        self,
        variable: SnapShotVariable,
    ) -> str:
        return repr(variable)

    @override
    def _compare_snapshot_variable(
        self,
        v1: SnapShotVariable,
        v2: SnapShotVariable,
    ) -> bool:
        return v1 == v2

    @override
    def _diff_snapshot_variables(
        self,
        snapshot: Dict[str, SnapShotVariable],
        variables: Dict[str, SnapShotVariable],
    ) -> Dict[str, SnapShotVariable]:
        return {
            k: v
            for k, v in variables.items()
            if k not in snapshot or not self._compare_snapshot_variable(snapshot[k], v)
        }


Tracer = ReprTracer


class DebugTracer:

    def __init__(
        self,
        dest_dir: _StrOrPath,
        includes: Optional[List[_StrOrPath]] = None,
        skip_types: Optional[List[_StrOrPath]] = None,
    ):
        self.depth = 1

        self.dest_dir: Path = Path(dest_dir)
        self.includes = [str(p) for p in includes] if includes else []
        self.skip_types = set(skip_types) if skip_types else set()

        self._all_types = set()
        self._skip_types = set()
        self._remain_types = set()

        self._thread_lock = threading.Lock()

        self._serializer = Serializer(backend="pickle")
        self._representer = Representer()

    def __enter__(
        self,
    ):
        threading.settrace_all_threads(self.trace)
        return self

    def trace(self, frame: FrameType, event, arg):

        file_path = frame.f_code.co_filename
        if not any(
            file_path.startswith(include_path) for include_path in self.includes
        ):
            return self.trace

        for key, val in frame.f_locals.items():

            type_full_name = get_type_full_name(type(val))

            with self._thread_lock:
                self._all_types.add(type_full_name)
                
            if any(
                type_full_name.startswith(skip_pattern)
                for skip_pattern in self.skip_types
            ):

                repr = safe_repr(val)
                bytes = self._serializer.dumps(repr)

                with self._thread_lock:
                    self._skip_types.add(type_full_name)
            else:

                repr = self._representer.repr(val)
                try:
                    bytes = self._serializer.dumps(val)
                except:
                    bytes = self._serializer.dumps(repr)

                with self._thread_lock:
                    self._remain_types.add(type_full_name)

        return self.trace

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        threading.settrace_all_threads(None)

        self.dest_dir.mkdir(parents=True, exist_ok=True)

        with open(
            Path(self.dest_dir)
            / f"type_debugger__ppid-{os.getppid()}__pid-{os.getpid()}.json",
            "w",
        ) as f:
            json.dump(
                {
                    "remain_types": sorted(list(self._remain_types)),
                    "skip_types": sorted(list(self._skip_types)),
                    "all_types": sorted(list(self._all_types)),
                },
                f,
                indent=4,
            )
