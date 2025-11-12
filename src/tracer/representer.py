from abc import ABC, abstractmethod
from inspect import isclass
import io
import reprlib
from typing import Any, Callable, Dict, Optional, Tuple, Union
from typing_extensions import override
import re
import numpy as np

from rich.tree import Tree
from rich.console import Console
from rich.pretty import Pretty
from rich.pretty import _is_attr_object, _is_namedtuple, is_dataclass



from tracer.types import *


def safe_repr(
    val: Any,
) -> str:
    try:
        return repr(val)
    except Exception as e:
        return f"<unrepresentable object {type(val).__name__} of {str(e)}>"


def safe_str(
    val: Any,
) -> str:
    try:
        return str(val)
    except Exception as e:
        return f"<unstringable object {type(val).__name__} of {str(e)}>"


class AbstractRepresenter(ABC):
    @abstractmethod
    def repr(
        self,
        obj: Any,
    ) -> str: ...


class SafeReprRepresenter(AbstractRepresenter):
    @override
    def repr(
        self,
        obj: Any,
    ) -> str:
        return safe_repr(obj)


class ReprLibRepresenter(AbstractRepresenter):
    def __init__(
        self,
        max_depth: int = 3,
        max_length: int = 50,
        max_string: int = 200,
    ):
        self._repr = reprlib.Repr(
            maxlevel=max_depth,
            maxdict=max_length,
            maxlist=max_length,
            maxtuple=max_length,
            maxset=max_length,
            maxfrozenset=max_length,
            maxdeque=max_length,
            maxstring=max_string,
        )

    @override
    def repr(
        self,
        obj: Any,
    ) -> str:
        return self._repr.repr(obj)


class RichRepresenter(AbstractRepresenter):
    def __init__(
        self,
        max_depth: int = 3,
        max_length: int = 20,
        max_string: int = 100,
    ):
        self._max_depth = max_depth
        self._max_length = max_length
        self._max_string = max_string

    @override
    def repr(
        self,
        obj: Any,
    ) -> str:
        buf = io.StringIO()
        console = Console(file=buf)
        console.print(
            Pretty(
                obj,
                max_depth=self._max_depth,
                max_length=self._max_length,
                max_string=self._max_string,
            )
        )
        return buf.getvalue().strip()


class NumpyArrayListRepresenter(AbstractRepresenter):
    @override
    def repr(
        self,
        obj: Any,
    ) -> str:
        result = repr(np.array(obj))
        result = re.sub(r"^array", "builtins.list", result)
        return result


class AbstractRegistryRepresenter(AbstractRepresenter):
    def __init__(self):
        self._type_helpers: Dict[Tuple[type, ...], AbstractRepresenter] = {}
        self._cond_helpers: Dict[Tuple[Callable, ...], AbstractRepresenter] = {}
        self.setup()

    def register(
        self,
        key: Union[
            type,
            Tuple[type, ...],
            Callable[[bool], Any],
            Tuple[Callable[[bool], Any], ...],
        ],
        helper: AbstractRepresenter,
    ):
        if not isinstance(key, tuple):
            key = (key,)
        if all(isclass(k) for k in key):
            self._type_helpers[key] = helper
            return
        if all(callable(k) for k in key):
            self._cond_helpers[key] = helper
            return
        raise RuntimeError("Mixed type and callable in register key is not supported.")

    def unregister(
        self,
        key: Union[
            type,
            Tuple[type, ...],
            Callable[[bool], Any],
            Tuple[Callable[[bool], Any], ...],
        ],
    ):
        if not isinstance(key, tuple):
            key = (key,)
        if key in self._type_helpers:
            del self._type_helpers[key]
            return
        if key in self._cond_helpers:
            del self._cond_helpers[key]
            return

    def helper(
        self,
        obj: Any,
    ) -> Optional[AbstractRepresenter]:
        try:
            for types, helper in reversed(self._type_helpers.items()):
                if issubclass(type(obj), types):
                    return helper
            for conds, helper in reversed(self._cond_helpers.items()):
                if any(cond(obj) for cond in conds):
                    return helper
            return None
        except:
            return None

    @abstractmethod
    def repr(
        self,
        obj: Any,
    ) -> str: ...

    @abstractmethod
    def setup(self) -> None: ...


class SafeRichRegistryRepresenter(AbstractRegistryRepresenter):
    def setup(self):
        rrh = RichRepresenter()
        self.register(
            CollectionTypes,
            rrh,
        )
        self.register(
            StringTypes,
            rrh,
        )
        self.register(
            PrimitiveType,
            rrh,
        )
        self.register(
            (
                lambda o: hasattr(o, "__rich_repr__") and not isclass(o),
                _is_attr_object,
                is_dataclass,
                _is_namedtuple,
            ),
            rrh,
        )

    def _repr(
        self,
        obj: Any,
        tree: Tree,
        depth: int,
        max_depth: int,
        visited: set,
    ) -> str:
        cls = type(obj)

        if helper := self.helper(obj):
            tree.add(helper.repr(obj))
            return

        if cls.__repr__ is not object.__repr__:
            tree.add(safe_repr(obj))
            return

        if cls.__str__ is not object.__str__:
            tree.add(safe_str(obj))
            return

        obj_id = id(obj)
        if obj_id in visited:
            tree.add(f"... (Cycle detected: {type(obj).__name__})")
            return
        visited.add(obj_id)

        if depth >= max_depth:
            tree.add("... (Max depth reached)")
            return

        try:
            attrs = []
            if hasattr(obj, "__dict__"):
                try:
                    for name, value in vars(obj).items():
                        attrs.append((name, value))
                except:
                    pass
            if hasattr(obj, "__slots__"):
                try:
                    for name in obj.__slots__:
                        if any(name == k for k, v in attrs):
                            continue
                        try:
                            value = getattr(obj, name)
                            attrs.append((name, value))
                        except AttributeError:
                            attrs.append((name, "<not set>"))
                except:
                    pass
            if not attrs:
                tree.add(safe_repr(obj))
                return

            attrs.sort()

            for name, value in attrs:
                self._repr(
                    obj=value,
                    tree=tree.add(f"{name}: "),
                    depth=depth + 1,
                    max_depth=max_depth,
                    visited=visited.copy(),
                )

        finally:
            visited.remove(obj_id)

    @override
    def repr(
        self,
        obj: Any,
        max_depth: int = 3,
    ) -> str:
        try:
            if obj is None:
                return "None"

            cls = type(obj)
            if helper := self.helper(obj):
                return helper.repr(obj)
            if cls.__repr__ is not object.__repr__:
                return safe_repr(obj)
            if cls.__str__ is not object.__str__:
                return safe_str(obj)

            buf = io.StringIO()
            console = Console(file=buf)
            tree = Tree(f"{type(obj).__name__} at {hex(id(obj))}")
            self._repr(
                obj=obj,
                tree=tree,
                depth=0,
                max_depth=max_depth,
                visited=set(),
            )
            console.print(tree)
            result = buf.getvalue().strip()
            try:
                if cls in CollectionTypes and hasattr(cls, "__len__"):
                    result += f"\n(length: {len(obj)})"
            except:
                pass
            try:
                if hasattr(cls, "__bases__"):
                    bases = [base.__name__ for base in cls.__bases__]
                    if bases:
                        result += f"\n(bases: {', '.join(bases)})"
            except:
                pass
            return result

        # IF REPR IS FAILED
        except:
            return safe_repr(obj)


Representer = SafeRichRegistryRepresenter
