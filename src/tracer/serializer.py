import os
import tempfile
from typing import Any, Callable, Dict, Literal, Tuple

import io

import dill
import joblib
import pickle


def _joblib_dumps(
    data: Any,
) -> bytes:

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_filename = tmp_file.name

    try:
        joblib.dump(data, tmp_filename, compress=0)
        with open(tmp_filename, "rb") as f:
            serialized_data = f.read()
        return serialized_data

    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)


def _joblib_loads(
    data: bytes,
) -> Any:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_filename = tmp_file.name
        tmp_file.write(data)

    try:
        loaded_object = joblib.load(tmp_filename)
        return loaded_object

    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)


class Serializer:

    BACKENDS: Dict[
        Literal["pickle", "dill", "joblib"],
        Tuple[Callable[..., bytes], Callable[[bytes], Any]],
    ] = {
        "pickle": (pickle.dumps, pickle.loads),
        "dill": (dill.dumps, dill.loads),
        "joblib": (_joblib_dumps, _joblib_loads),
    }

    def __init__(
        self,
        backend: Literal["pickle", "dill", "joblib"] = "pickle",
    ):

        if backend not in Serializer.BACKENDS:
            raise ValueError(f"Unsupported backend: {backend}")

        self._backend = backend
        self._dumps, self._loads = Serializer.BACKENDS[backend]

    @property
    def backend(self) -> str:
        return self._backend

    def dumps(self, data: Any, *args, **kwargs) -> bytes:
        return self._dumps(data, *args, **kwargs)

    def loads(self, data: bytes, *args, **kwargs) -> Any:
        return self._loads(data, *args, **kwargs)
