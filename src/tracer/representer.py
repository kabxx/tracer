from abc import ABC, abstractmethod
from inspect import isclass
import io
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union, Set
from typing_extensions import override

from rich.tree import Tree
from rich.console import Console
from rich.pretty import Pretty
from rich.pretty import _is_attr_object, _is_namedtuple, is_dataclass


from array import array
from collections import deque
import os


from collections import defaultdict, Counter, UserDict, UserList
from types import MappingProxyType
from typing import Optional

CollectionTypes = (
    os._Environ,
    array,
    defaultdict,
    Counter,
    deque,
    dict,
    UserDict,
    frozenset,
    list,
    UserList,
    set,
    tuple,
    MappingProxyType,
)

StringTypes = (
    str,
    bytes,
    bytearray,
)

PrimitiveType = (
    bool,
    int,
    float,
    complex,
    type(None),
)


def get_full_qualified_name(
    obj: Any,
) -> Optional[str]:
    try:
        cls = type(obj)
        return cls.__module__ + "." + cls.__qualname__
    except:
        return None


def _should_obj_skip(
    obj: Any,
    skip_types: Optional[Set[str]] = None,
) -> bool:
    if skip_types is None:
        return False
    full_qualified_name = get_full_qualified_name(obj)
    return any(full_qualified_name.startswith(p) for p in skip_types)


def safe_repr(
    val: Any,
) -> str:
    try:
        cls = type(val)
        return f"<{cls.__module__}.{cls.__name__} object at {hex(id(val))}>"
    except Exception as e:
        return f"<unrepresentable object {type(val).__name__} of {str(e)}>"


def no_raise_repr(
    val: Any,
) -> str:
    try:
        return repr(val)
    except Exception as e:
        return f"<unrepresentable object {type(val).__name__} of {str(e)}>"


def no_raise_str(
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
        skip_types: Optional[Set[str]] = None,
    ) -> str: ...


class SafeRepresenter(AbstractRepresenter):
    @override
    def repr(
        self,
        obj: Any,
        skip_types: Optional[Set[str]] = None,
    ) -> str:
        return safe_repr(obj)


class NoRaiseRepresenter(AbstractRepresenter):
    @override
    def repr(
        self,
        obj: Any,
        skip_types: Optional[Set[str]] = None,
    ) -> str:
        if _should_obj_skip(obj, skip_types=skip_types):
            return safe_repr(obj)
        return no_raise_repr(obj)


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
        skip_types: Optional[Set[str]] = None,
    ) -> str:
        buf = io.StringIO()
        console = Console(file=buf)
        console.print(
            RichPretty(
                obj,
                max_depth=self._max_depth,
                max_length=self._max_length,
                max_string=self._max_string,
                skip_types=skip_types,
            )
        )
        return buf.getvalue().strip()


class AbstractRegistryRepresenter(AbstractRepresenter):
    def __init__(self):
        self._type_representers: Dict[Tuple[type, ...], AbstractRepresenter] = {}
        self._cond_representers: Dict[Tuple[Callable, ...], AbstractRepresenter] = {}
        self._setup()

    def register(
        self,
        key: Union[
            type,
            Tuple[type, ...],
            Callable[[bool], Any],
            Tuple[Callable[[bool], Any], ...],
        ],
        representer: AbstractRepresenter,
    ):
        if not isinstance(key, tuple):
            key = (key,)
        if all(isclass(k) for k in key):
            self._type_representers[key] = representer
            return
        if all(callable(k) for k in key):
            self._cond_representers[key] = representer
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
        if key in self._type_representers:
            del self._type_representers[key]
            return
        if key in self._cond_representers:
            del self._cond_representers[key]
            return

    def get_representer(
        self,
        obj: Any,
    ) -> Optional[AbstractRepresenter]:
        try:
            for types, representer in reversed(self._type_representers.items()):
                if issubclass(type(obj), types):
                    return representer
            for conds, representer in reversed(self._cond_representers.items()):
                if any(cond(obj) for cond in conds):
                    return representer
            return None
        except:
            return None

    @abstractmethod
    def repr(
        self,
        obj: Any,
        skip_types: Optional[Set[str]] = None,
    ) -> str: ...

    @abstractmethod
    def _setup(self) -> None: ...


class NoRaiseRichRegistryRepresenter(AbstractRegistryRepresenter):
    def _setup(self):
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
        skip_types: Optional[Set[str]],
    ) -> str:
        
        # check skip types
        if _should_obj_skip(obj, skip_types=skip_types):
            tree.add(safe_repr(obj))
            return
        
        cls = type(obj)

        # use representer
        if representer := self.get_representer(obj):
            tree.add(
                representer.repr(
                    obj,
                    skip_types=skip_types,
                )
            )
            return
        
        # use default repr or str
        if cls.__repr__ is not object.__repr__:
            tree.add(no_raise_repr(obj))
            return
        if cls.__str__ is not object.__str__:
            tree.add(no_raise_str(obj))
            return
        
        # make repr with rich tree
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
                tree.add(no_raise_repr(obj))
                return
            attrs.sort()
            for name, value in attrs:
                self._repr(
                    obj=value,
                    tree=tree.add(f"{name}: "),
                    depth=depth + 1,
                    max_depth=max_depth,
                    visited=visited.copy(),
                    skip_types=skip_types,
                )

        finally:
            visited.remove(obj_id)

    @override
    def repr(
        self,
        obj: Any,
        skip_types: Optional[Set[str]] = None,
        max_depth: int = 3,
    ) -> str:

        # check None
        if obj is None:
            return "None"
        
        try:
        
            # check skip types
            if _should_obj_skip(obj, skip_types=skip_types):
                return safe_repr(obj)
        
            # use representer
            cls = type(obj)
            if representer := self.get_representer(obj):
                return representer.repr(
                    obj,
                    skip_types=skip_types,
                )
        
            # use default repr or str
            if cls.__repr__ is not object.__repr__:
                return no_raise_repr(
                    obj,
                )
            if cls.__str__ is not object.__str__:
                return no_raise_str(
                    obj,
                )
        
            # make repr with rich tree
            buf = io.StringIO()
            console = Console(file=buf)
            tree = Tree(f"{type(obj).__name__} at {hex(id(obj))}")
            self._repr(
                obj=obj,
                tree=tree,
                depth=0,
                max_depth=max_depth,
                visited=set(),
                skip_types=skip_types,
            )
            console.print(tree)
            result = buf.getvalue().strip()

            # append object meta information
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
        except Exception as e:
            return no_raise_repr(obj)


Representer = NoRaiseRichRegistryRepresenter

### Overwrite rich.pretty ###

from dataclasses import fields, is_dataclass
from inspect import isclass
from itertools import islice
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from rich.repr import RichReprResult
from rich._loop import loop_last
from rich.text import Text
from rich._pick import pick_bool
from rich.measure import Measurement
from rich.cells import cell_len

from rich.pretty import (
    Node,
    _safe_isinstance,
    _is_attr_object,
    _get_attr_fields,
    _is_dataclass_repr,
    _has_default_namedtuple_repr,
    _is_namedtuple,
    _CONTAINERS,
    _MAPPING_CONTAINERS,
    _BRACES,
)

if TYPE_CHECKING:
    from rich.console import (
        Console,
        ConsoleOptions,
        RenderResult,
    )


def traverse(
    _object: Any,
    max_length: Optional[int] = None,
    max_string: Optional[int] = None,
    max_depth: Optional[int] = None,
    skip_types: Optional[Set[type]] = None,
) -> Node:
    """Traverse object and generate a tree.

    Args:
        _object (Any): Object to be traversed.
        max_length (int, optional): Maximum length of containers before abbreviating, or None for no abbreviation.
            Defaults to None.
        max_string (int, optional): Maximum length of string before truncating, or None to disable truncating.
            Defaults to None.
        max_depth (int, optional): Maximum depth of data structures, or None for no maximum.
            Defaults to None.

    Returns:
        Node: The root of a tree structure which can be used to render a pretty repr.
    """

    def to_repr(obj: Any) -> str:
        """Get repr string for an object, but catch errors."""

        skip_obj = _should_obj_skip(obj, skip_types=skip_types)

        if (
            max_string is not None
            and _safe_isinstance(obj, (bytes, str))
            and len(obj) > max_string
        ):
            truncated = len(obj) - max_string
            obj_repr = f"{obj[:max_string]!r}+{truncated}"
        else:
            if skip_obj:
                obj_repr = safe_repr(obj)
            else:
                try:
                    obj_repr = repr(obj)
                except Exception as error:
                    obj_repr = f"<repr-error {str(error)!r}>"
        return obj_repr

    visited_ids: Set[int] = set()
    push_visited = visited_ids.add
    pop_visited = visited_ids.remove

    def _traverse(obj: Any, root: bool = False, depth: int = 0) -> Node:
        """Walk the object depth first."""

        skip_obj = _should_obj_skip(obj, skip_types=skip_types)

        obj_id = id(obj)
        if obj_id in visited_ids:
            # Recursion detected
            return Node(value_repr="...")

        obj_type = type(obj)
        children: List[Node]
        reached_max_depth = max_depth is not None and depth >= max_depth

        def iter_rich_args(rich_args: Any) -> Iterable[Union[Any, Tuple[str, Any]]]:
            for arg in rich_args:
                if _safe_isinstance(arg, tuple):
                    if len(arg) == 3:
                        key, child, default = arg
                        if default == child:
                            continue
                        yield key, child
                    elif len(arg) == 2:
                        key, child = arg
                        yield key, child
                    elif len(arg) == 1:
                        yield arg[0]
                else:
                    yield arg

        if skip_obj:
            try:
                fake_attributes = hasattr(
                    obj, "awehoi234_wdfjwljet234_234wdfoijsdfmmnxpi492"
                )
            except Exception:
                fake_attributes = False
        else:
            fake_attributes = True

        rich_repr_result: Optional[RichReprResult] = None
        if not fake_attributes:
            try:
                if hasattr(obj, "__rich_repr__") and not isclass(obj):
                    rich_repr_result = obj.__rich_repr__()
            except Exception:
                pass

        if rich_repr_result is not None:
            push_visited(obj_id)
            angular = getattr(obj.__rich_repr__, "angular", False)
            args = list(iter_rich_args(rich_repr_result))
            class_name = obj.__class__.__name__

            if args:
                children = []
                append = children.append

                if reached_max_depth:
                    if angular:
                        node = Node(value_repr=f"<{class_name}...>")
                    else:
                        node = Node(value_repr=f"{class_name}(...)")
                else:
                    if angular:
                        node = Node(
                            open_brace=f"<{class_name} ",
                            close_brace=">",
                            children=children,
                            last=root,
                            separator=" ",
                        )
                    else:
                        node = Node(
                            open_brace=f"{class_name}(",
                            close_brace=")",
                            children=children,
                            last=root,
                        )
                    for last, arg in loop_last(args):
                        if _safe_isinstance(arg, tuple):
                            key, child = arg
                            child_node = _traverse(child, depth=depth + 1)
                            child_node.last = last
                            child_node.key_repr = key
                            child_node.key_separator = "="
                            append(child_node)
                        else:
                            child_node = _traverse(arg, depth=depth + 1)
                            child_node.last = last
                            append(child_node)
            else:
                node = Node(
                    value_repr=f"<{class_name}>" if angular else f"{class_name}()",
                    children=[],
                    last=root,
                )
            pop_visited(obj_id)
        elif not fake_attributes and _is_attr_object(obj):
            push_visited(obj_id)
            children = []
            append = children.append

            attr_fields = _get_attr_fields(obj)
            if attr_fields:
                if reached_max_depth:
                    node = Node(value_repr=f"{obj.__class__.__name__}(...)")
                else:
                    node = Node(
                        open_brace=f"{obj.__class__.__name__}(",
                        close_brace=")",
                        children=children,
                        last=root,
                    )

                    def iter_attrs() -> (
                        Iterable[Tuple[str, Any, Optional[Callable[[Any], str]]]]
                    ):
                        """Iterate over attr fields and values."""
                        for attr in attr_fields:
                            if attr.repr:
                                try:
                                    value = getattr(obj, attr.name)
                                except Exception as error:
                                    # Can happen, albeit rarely
                                    yield (attr.name, error, None)
                                else:
                                    yield (
                                        attr.name,
                                        value,
                                        attr.repr if callable(attr.repr) else None,
                                    )

                    for last, (name, value, repr_callable) in loop_last(iter_attrs()):
                        if repr_callable:
                            child_node = Node(value_repr=str(repr_callable(value)))
                        else:
                            child_node = _traverse(value, depth=depth + 1)
                        child_node.last = last
                        child_node.key_repr = name
                        child_node.key_separator = "="
                        append(child_node)
            else:
                node = Node(
                    value_repr=f"{obj.__class__.__name__}()", children=[], last=root
                )
            pop_visited(obj_id)
        elif (
            not fake_attributes
            and is_dataclass(obj)
            and not _safe_isinstance(obj, type)
            and _is_dataclass_repr(obj)
        ):
            push_visited(obj_id)
            children = []
            append = children.append
            if reached_max_depth:
                node = Node(value_repr=f"{obj.__class__.__name__}(...)")
            else:
                node = Node(
                    open_brace=f"{obj.__class__.__name__}(",
                    close_brace=")",
                    children=children,
                    last=root,
                    empty=f"{obj.__class__.__name__}()",
                )

                for last, field in loop_last(
                    field
                    for field in fields(obj)
                    if field.repr and hasattr(obj, field.name)
                ):
                    child_node = _traverse(getattr(obj, field.name), depth=depth + 1)
                    child_node.key_repr = field.name
                    child_node.last = last
                    child_node.key_separator = "="
                    append(child_node)

            pop_visited(obj_id)
        elif (
            not fake_attributes
            and _is_namedtuple(obj)
            and _has_default_namedtuple_repr(obj)
        ):
            push_visited(obj_id)
            class_name = obj.__class__.__name__
            if reached_max_depth:
                # If we've reached the max depth, we still show the class name, but not its contents
                node = Node(
                    value_repr=f"{class_name}(...)",
                )
            else:
                children = []
                append = children.append
                node = Node(
                    open_brace=f"{class_name}(",
                    close_brace=")",
                    children=children,
                    empty=f"{class_name}()",
                )
                for last, (key, value) in loop_last(obj._asdict().items()):
                    child_node = _traverse(value, depth=depth + 1)
                    child_node.key_repr = key
                    child_node.last = last
                    child_node.key_separator = "="
                    append(child_node)
            pop_visited(obj_id)
        elif not fake_attributes and _safe_isinstance(obj, _CONTAINERS):
            for container_type in _CONTAINERS:
                if _safe_isinstance(obj, container_type):
                    obj_type = container_type
                    break

            push_visited(obj_id)

            open_brace, close_brace, empty = _BRACES[obj_type](obj)

            if reached_max_depth:
                node = Node(value_repr=f"{open_brace}...{close_brace}")
            elif obj_type.__repr__ != type(obj).__repr__:
                node = Node(value_repr=to_repr(obj), last=root)
            elif obj:
                children = []
                node = Node(
                    open_brace=open_brace,
                    close_brace=close_brace,
                    children=children,
                    last=root,
                )
                append = children.append
                num_items = len(obj)
                last_item_index = num_items - 1

                if _safe_isinstance(obj, _MAPPING_CONTAINERS):
                    iter_items = iter(obj.items())
                    if max_length is not None:
                        iter_items = islice(iter_items, max_length)
                    for index, (key, child) in enumerate(iter_items):
                        child_node = _traverse(child, depth=depth + 1)
                        child_node.key_repr = to_repr(key)
                        child_node.last = index == last_item_index
                        append(child_node)
                else:
                    iter_values = iter(obj)
                    if max_length is not None:
                        iter_values = islice(iter_values, max_length)
                    for index, child in enumerate(iter_values):
                        child_node = _traverse(child, depth=depth + 1)
                        child_node.last = index == last_item_index
                        append(child_node)
                if max_length is not None and num_items > max_length:
                    append(Node(value_repr=f"... +{num_items - max_length}", last=True))
            else:
                node = Node(empty=empty, children=[], last=root)

            pop_visited(obj_id)
        else:
            node = Node(value_repr=to_repr(obj), last=root)
        node.is_tuple = type(obj) == tuple
        node.is_namedtuple = _is_namedtuple(obj)
        return node

    node = _traverse(_object, root=True)
    return node


def pretty_repr(
    _object: Any,
    *,
    max_width: int = 80,
    indent_size: int = 4,
    max_length: Optional[int] = None,
    max_string: Optional[int] = None,
    max_depth: Optional[int] = None,
    expand_all: bool = False,
    skip_types: Optional[Set[str]] = None,
) -> str:
    """Prettify repr string by expanding on to new lines to fit within a given width.

    Args:
        _object (Any): Object to repr.
        max_width (int, optional): Desired maximum width of repr string. Defaults to 80.
        indent_size (int, optional): Number of spaces to indent. Defaults to 4.
        max_length (int, optional): Maximum length of containers before abbreviating, or None for no abbreviation.
            Defaults to None.
        max_string (int, optional): Maximum length of string before truncating, or None to disable truncating.
            Defaults to None.
        max_depth (int, optional): Maximum depth of nested data structure, or None for no depth.
            Defaults to None.
        expand_all (bool, optional): Expand all containers regardless of available width. Defaults to False.

    Returns:
        str: A possibly multi-line representation of the object.
    """

    if _safe_isinstance(_object, Node):
        node = _object
    else:
        node = traverse(
            _object,
            max_length=max_length,
            max_string=max_string,
            max_depth=max_depth,
            skip_types=skip_types,
        )
    repr_str: str = node.render(
        max_width=max_width, indent_size=indent_size, expand_all=expand_all
    )
    return repr_str


class RichPretty(Pretty):

    def __init__(
        self,
        *args: Any,
        skip_types: Optional[Set[str]] = None,
        **kwds: Any,
    ) -> None:
        super().__init__(*args, **kwds)
        self.skip_types = skip_types

    def __rich_console__(
        self,
        console: "Console",
        options: "ConsoleOptions",
    ) -> "RenderResult":
        pretty_str = pretty_repr(
            self._object,
            max_width=options.max_width - self.margin,
            indent_size=self.indent_size,
            max_length=self.max_length,
            max_string=self.max_string,
            max_depth=self.max_depth,
            expand_all=self.expand_all,
            skip_types=self.skip_types,
        )
        pretty_text = Text.from_ansi(
            pretty_str,
            justify=self.justify or options.justify,
            overflow=self.overflow or options.overflow,
            no_wrap=pick_bool(self.no_wrap, options.no_wrap),
            style="pretty",
        )
        pretty_text = (
            self.highlighter(pretty_text)
            if pretty_text
            else Text(
                f"{type(self._object)}.__repr__ returned empty string",
                style="dim italic",
            )
        )
        if self.indent_guides and not options.ascii_only:
            pretty_text = pretty_text.with_indent_guides(
                self.indent_size, style="repr.indent"
            )
        if self.insert_line and "\n" in pretty_text:
            yield ""
        yield pretty_text

    def __rich_measure__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "Measurement":
        pretty_str = pretty_repr(
            self._object,
            max_width=options.max_width,
            indent_size=self.indent_size,
            max_length=self.max_length,
            max_string=self.max_string,
            max_depth=self.max_depth,
            expand_all=self.expand_all,
            skip_types=self.skip_types,
        )
        text_width = (
            max(cell_len(line) for line in pretty_str.splitlines()) if pretty_str else 0
        )
        return Measurement(text_width, text_width)
