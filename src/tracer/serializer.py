import os
import tempfile
from typing import Any, Callable, Dict, Literal, Tuple

import dill
import pickle


class Serializer:

    BACKENDS: Dict[
        Literal["pickle", "dill"],
        Tuple[Callable[..., bytes], Callable[[bytes], Any]],
    ] = {
        "pickle": (pickle.dumps, pickle.loads),
        "dill": (dill.dumps, dill.loads),
    }

    def __init__(
        self,
        backend: Literal["pickle", "dill"] = "pickle",
    ):

        if backend not in Serializer.BACKENDS:
            raise ValueError(f"Unsupported backend: {backend}")

        self._backend = backend
        self._dumps, self._loads = Serializer.BACKENDS[backend]

    @property
    def backend(self) -> str:
        return self._backend

    def dumps(
        self,
        data: Any,
        *args,
        **kwds,
    ) -> bytes:
        return self._dumps(data, *args, **kwds)

    def loads(
        self,
        data: bytes,
        *args,
        **kwds,
    ) -> Any:
        return self._loads(data, *args, **kwds)
