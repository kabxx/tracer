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
    Set,
    Tuple,
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
    get_full_qualified_name,
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
        trace_locals: bool = True,
        trace_globals: bool = False,
        only_trace_changed_locals: bool = True,
        only_trace_changed_globals: bool = True,
        skip_empty_trace: bool = True,
        only_main_thread: bool = False,
        skip_empty_thread: bool = True,
        only_main_process: bool = False,
        *,
        interval: Optional[
            Tuple[
                Optional[Tuple[Optional[str], str]], Optional[Tuple[Optional[str], str]]
            ]
        ] = None,
        includes: Optional[List[_StrOrPath]] = None,
        excludes: Optional[List[_StrOrPath]] = None,
        targets: Optional[List[Tuple[str, str]]] = None,
    ):
        kwds = locals()
        kwds.pop("self")
        self._kwds = kwds.copy()

        self._status: BaseTracer.Status = BaseTracer.Status.INIT

        self._dest_dir = Path(dest_dir)
        if self._dest_dir.exists() and not self._dest_dir.is_dir():
            raise ValueError(f"Destination {dest_dir} is not a directory.")

        # basic
        self._trace_locals = trace_locals
        self._trace_globals = trace_globals
        self._only_trace_changed_locals = only_trace_changed_locals
        self._only_trace_changed_globals = only_trace_changed_globals
        self._skip_empty_trace = skip_empty_trace

        # thread
        self._only_main_thread = only_main_thread
        self._skip_empty_thread = skip_empty_thread
        self._thread_lock = threading.Lock()
        self._thread_local = threading.local()
        self._thread_contexts: List[TracerThreadContext] = []

        # process
        self._only_main_process = only_main_process
        self._hook_context: HookContext = HookContext()

        # features

        # interval
        if interval is not None:
            beg, end = interval
            if beg is None:
                (
                    self._interval_beg,
                    self._interval_beg_ok,
                ) = (None, None)
            else:
                (
                    self._interval_beg,
                    self._interval_beg_ok,
                ) = (beg, False)
            if end is None:
                (
                    self._interval_end,
                    self._interval_end_ok,
                ) = (None, None)
            else:
                (
                    self._interval_end,
                    self._interval_end_ok,
                ) = (end, False)
        else:
            (
                self._interval_beg,
                self._interval_beg_ok,
            ) = (None, None)
            (
                self._interval_end,
                self._interval_end_ok,
            ) = (None, None)

        # includes or excludes
        self._includes = [str(p) for p in includes] if includes else None
        self._excludes = [str(p) for p in excludes] if excludes else None

        # targets
        self._targets: Set[Tuple[str, str, int]] = (
            set(targets) if targets is not None else None
        )

    def _check_includes_or_excludes(
        self,
        file_path: str,
    ) -> bool:
        if self._includes is not None:
            if not any(file_path.startswith(pattern) for pattern in self._includes):
                return False
        if self._excludes is not None:
            if any(file_path.startswith(pattern) for pattern in self._excludes):
                return False
        return True

    def _check_interval(
        self,
        event: str,
        file_path: str,
        func_name: str,
    ) -> bool:
        if self._interval_beg:
            if self._interval_beg_ok:
                return True
            beg_path, beg_func = self._interval_beg
            if event == "return" and beg_func == func_name:
                if not beg_path or file_path == beg_path:
                    self._interval_beg_ok = True
            else:
                return False
        if self._interval_end:
            if self._interval_end_ok:
                return False
            end_path, end_func = self._interval_end
            if event == "call" and end_func == func_name:
                if not end_path or file_path == end_path:
                    self._interval_end_ok = True
                    return False
            else:
                return True
        return True

    def _check_targets(
        self,
        event: str,
        file_path: str,
        func_name: str,
    ) -> bool:
        if self._targets:
            targets: Dict = self._store("targets")
            target = (file_path, func_name)
            # track function call
            if event == "call":
                if target in self._targets:
                    if target in targets:
                        targets[target] += 1
                    else:
                        targets[target] = 1
            try:
                if not targets:
                    return False
            finally:
                # track function return
                if event == "return":
                    if target in self._targets:
                        if target in targets:
                            if targets[target] > 1:
                                targets[target] -= 1
                            else:
                                del targets[target]
        return True

    def trace(
        self,
        frame: FrameType,
        event: str,
        arg: Any,
    ) -> Optional[Callable[[FrameType, str, Any], Any]]:

        # update depth
        if event == "call":
            self._depth_set(self._depth_get() + 1)
        elif event == "return":
            self._depth_set(self._depth_get() - 1)

        file_path = frame.f_code.co_filename
        func_name = frame.f_code.co_name
        func_line = frame.f_lineno

        if not self._check_interval(event, file_path, func_name):
            return self.trace

        if not self._check_includes_or_excludes(file_path):
            return self.trace

        if not self._check_targets(event, file_path, func_name):
            return self.trace

        if event == "call":
            locals = self._snapshot_variables(frame.f_locals)
            # create function trace
            function: FunctionDict = {
                "file": file_path,
                "depth": self._depth_get(),
                "name": func_name,
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
            self._stack().append(function)

        elif event == "return":

            # pop function
            function = self._stack().pop()
            function["retval"] = self._output_from_snapshot_variable(
                self._snapshot_variable(arg)
            )

            # ensure trace available
            self._caller_fix(
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
            self._caller()["traces"][-1]["functions"].append(function)

        elif event == "line":

            trace = {
                "line": func_line,
                "functions": [],
                "exceptions": [],
            }

            # make trace locals
            if self._trace_locals:
                locals = self._snapshot_variables(frame.f_locals)
                if self._only_trace_changed_locals:
                    locals = self._diff_snapshot_variables(
                        self._caller()["__local_snapshot__"],
                        locals,
                    )
                    self._caller()["__local_snapshot__"].update(locals)
                trace["locals"] = self._output_from_snapshot_variables(locals)

            # make trace globales
            if self._trace_globals:
                globals = self._snapshot_variables(frame.f_globals)
                if self._only_trace_changed_globals:
                    globals = self._diff_snapshot_variables(
                        self._caller()["__global_snapshot__"],
                        globals,
                    )
                    self._caller()["__global_snapshot__"].update(globals)
                trace["globals"] = self._output_from_snapshot_variables(globals)

            # append trace
            if not self._skip_empty_trace or not self._is_function_trace_empty(trace):
                self._caller()["traces"].append(trace)

        elif event == "exception":
            # ensure trace available
            self._caller_fix(
                frame=frame,
            )

            exc_type, exc_value, exc_traceback = arg
            self._caller()["traces"][-1]["exceptions"].append(
                {
                    "function": func_name,
                    "type": self._output_by_exc_type(exc_type),
                    "value": self._output_by_exc_val(exc_value),
                    "traceback": self._output_by_exc_traceback(exc_traceback),
                }
            )

        return self.trace

    def _store(
        self,
        key: str,
    ) -> Dict:
        try:
            store = getattr(self._thread_local, "store")
        except AttributeError:
            store = {}
            setattr(self._thread_local, "store", store)
        try:
            value = store[key]
        except KeyError:
            value = {}
            store[key] = value
        return value

    def _context(
        self,
    ) -> TracerThreadContext:
        if hasattr(self._thread_local, "context"):
            return getattr(self._thread_local, "context")
        thread_context = TracerThreadContext(thread=threading.current_thread())
        setattr(self._thread_local, "context", thread_context)
        with self._thread_lock:
            self._thread_contexts.append(thread_context)
        return thread_context

    def _depth_get(
        self,
    ) -> int:
        return self._context()._depth

    def _depth_set(
        self,
        depth: int,
    ) -> None:
        self._context()._depth = depth

    def _stack(
        self,
    ) -> List[FunctionDict]:
        return self._context()._stacks

    def _caller(
        self,
    ) -> FunctionDict:
        return self._stack()[-1]

    def _caller_fix(
        self,
        frame: FrameType,
    ) -> None:
        if self._caller()["traces"]:
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
                    self._caller()["__local_snapshot__"], locals
                )
                self._caller()["__local_snapshot__"].update(locals)
            trace["locals"] = self._output_from_snapshot_variables(locals)

        if self._trace_globals:
            globals = self._snapshot_variables(frame.f_globals)
            if self._only_trace_changed_globals:
                globals = self._diff_snapshot_variables(
                    self._caller()["__global_snapshot__"],
                    globals,
                )
                self._caller()["__global_snapshot__"].update(globals)
            trace["globals"] = self._output_from_snapshot_variables(globals)

        self._caller()["traces"].append(trace)

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
            import multiprocess.process  # type: ignore

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
                    # import jsonpickle

                    # f.write(jsonpickle.encode(trace, unpicklable=False, indent=4))
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
        trace_locals: bool = True,
        trace_globals: bool = False,
        only_trace_changed_locals: bool = True,
        only_trace_changed_globals: bool = True,
        skip_empty_trace: bool = True,
        only_main_thread: bool = False,
        skip_empty_thread: bool = True,
        only_main_process: bool = False,
        serializer_backend: Literal["pickle", "dill"] = "pickle",
        skip_repr_types: Optional[List[str]] = None,
        skip_copy_types: Optional[List[str]] = None,
        *,
        interval: Optional[
            Tuple[
                Optional[Tuple[Optional[str], str]], Optional[Tuple[Optional[str], str]]
            ]
        ] = None,
        includes: Optional[List[_StrOrPath]] = None,
        excludes: Optional[List[_StrOrPath]] = None,
        targets: Optional[List[Tuple[str, str, int]]] = None,
    ):
        super().__init__(
            dest_dir=dest_dir,
            trace_locals=trace_locals,
            trace_globals=trace_globals,
            only_trace_changed_locals=only_trace_changed_locals,
            only_trace_changed_globals=only_trace_changed_globals,
            skip_empty_trace=skip_empty_trace,
            only_main_thread=only_main_thread,
            skip_empty_thread=skip_empty_thread,
            only_main_process=only_main_process,
            interval=interval,
            includes=includes,
            excludes=excludes,
            targets=targets,
        )

        self._skip_copy_types = set(skip_copy_types) if skip_copy_types else set()
        self._skip_repr_types = set(skip_repr_types) if skip_repr_types else set()

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
        return self.representer().repr(
            var,
            skip_types=self._skip_repr_types,
        )

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
        cls_full_name = get_full_qualified_name(variable)

        # skip repr
        if any(cls_full_name.startswith(p) for p in self._skip_repr_types):
            repr_str = safe_repr(variable)
            return self.SnapShotVariable(cls, repr_str, self.serialize(repr_str))

        # skip copy
        if any(cls_full_name.startswith(p) for p in self._skip_copy_types):
            repr_str = self.repr(variable)
            return self.SnapShotVariable(cls, repr_str, self.serialize(repr_str))

        # normal
        repr_str = self.repr(variable)
        try:
            copy_bytes = self.serialize(variable)
        except:
            copy_bytes = self.serialize(repr_str)
        return self.SnapShotVariable(cls, repr_str, copy_bytes)

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


class DebugTracer(BaseTracer):

    def __init__(
        self,
        dest_dir: _StrOrPath,
        dest_name: Optional[str] = None,
        skip_copy_types: Optional[List[str]] = None,
        skip_repr_types: Optional[List[str]] = None,
        only_main_thread: bool = False,
        *,
        interval: Optional[
            Tuple[
                Optional[Tuple[Optional[str], str]], Optional[Tuple[Optional[str], str]]
            ]
        ] = None,
        includes: Optional[List[_StrOrPath]] = None,
        excludes: Optional[List[_StrOrPath]] = None,
        targets: Optional[List[Tuple[str, str]]] = None,
    ):
        self.dest_dir: Path = Path(dest_dir)
        self._dest_name = dest_name or "types.json"

        self._serializer = Serializer(backend="pickle")
        self._representer = Representer()

        self._skip_copy_types = set(skip_copy_types) if skip_copy_types else set()
        self._skip_repr_types = set(skip_repr_types) if skip_repr_types else set()

        self._only_main_thread = only_main_thread

        self._outputs = {
            "skip_repr": set(),
            "skip_copy": set(),
            "remain": set(),
        }
        self._lock = threading.Lock()

        self._thread_local = threading.local()

        # features

        # interval
        if interval is not None:
            beg, end = interval
            if beg is None:
                (
                    self._interval_beg,
                    self._interval_beg_ok,
                ) = (None, None)
            else:
                (
                    self._interval_beg,
                    self._interval_beg_ok,
                ) = (beg, False)
            if end is None:
                (
                    self._interval_end,
                    self._interval_end_ok,
                ) = (None, None)
            else:
                (
                    self._interval_end,
                    self._interval_end_ok,
                ) = (end, False)
        else:
            (
                self._interval_beg,
                self._interval_beg_ok,
            ) = (None, None)
            (
                self._interval_end,
                self._interval_end_ok,
            ) = (None, None)

        # includes or excludes
        self._includes = [str(p) for p in includes] if includes else []
        self._excludes = [str(p) for p in excludes] if excludes else []

        self._targets: Set[Tuple[str, str, int]] = set(targets) if targets else None

    def serialize(
        self,
        val: Any,
    ) -> bytes:
        return self._serializer.dumps(val)

    def represent(
        self,
        val: Any,
    ) -> str:
        return self._representer.repr(
            val,
            skip_types=self._skip_repr_types,
        )

    def __enter__(
        self,
    ):
        if not self._only_main_thread:
            threading.settrace_all_threads(self.trace)
        else:
            sys.settrace(self.trace)
        return self

    def trace(
        self,
        frame: FrameType,
        event: str,
        arg: Any,
    ):

        file_path = frame.f_code.co_filename
        func_name = frame.f_code.co_name
        func_line = frame.f_lineno

        if not self._check_interval(event, file_path, func_name):
            return self.trace

        if not self._check_targets(event, file_path, func_name):
            return self.trace

        if not self._check_includes_or_excludes(file_path):
            return self.trace

        # if event == "call":
        #     print(func_name,file_path,func_line)

        if event == "call":
            print(
                f"TRACE CALL {frame.f_lineno} {frame.f_code.co_name} IN {frame.f_code.co_filename}"
            )

        for val in frame.f_locals.values():

            type_full_name = get_full_qualified_name(val)

            # print(type_full_name)

            if any(type_full_name.startswith(p) for p in self._skip_repr_types):
                # test
                self.serialize(safe_repr(val))
                with self._lock:
                    self._outputs["skip_repr"].add(type_full_name)
                continue

            if any(type_full_name.startswith(p) for p in self._skip_copy_types):
                # test
                self.serialize(self.represent(val))
                with self._lock:
                    self._outputs["skip_copy"].add(type_full_name)
                continue

            if True:
                # test
                try:
                    self.serialize(val)
                    self.represent(val)
                except:
                    self.serialize(self.represent(val))

                with self._lock:
                    self._outputs["remain"].add(type_full_name)
                continue

        return self.trace

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        if not self._only_main_thread:
            threading.settrace_all_threads(None)
        else:
            sys.settrace(None)

        self.dest_dir.mkdir(parents=True, exist_ok=True)

        with open(Path(self.dest_dir) / self._dest_name, "w") as f:
            json.dump(
                {
                    "skip_repr": sorted(self._outputs["skip_repr"]),
                    "skip_copy": sorted(self._outputs["skip_copy"]),
                    "remain": sorted(self._outputs["remain"]),
                },
                f,
                indent=4,
            )

    @override
    def _compare_snapshot_variable(self, v1, v2): ...

    @override
    def _diff_snapshot_variables(self, snapshot, variables): ...

    @override
    def _output_from_snapshot_variable(self, variable): ...

    @override
    def _output_by_exc_traceback(self, exception_traceback: TracebackType) -> str: ...

    @override
    def _output_by_exc_type(self, exception_type: Type[BaseException]) -> str: ...

    @override
    def _output_by_exc_val(self, exception_value: BaseException) -> str: ...

    @override
    def _snapshot_variable(self, variable): ...
