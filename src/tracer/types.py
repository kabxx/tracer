from array import array
from collections import deque
import os


from collections import defaultdict, Counter, UserDict, UserList
from types import MappingProxyType
from typing import Optional, Type

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

def get_type_full_name(
    cls: Type,
) -> Optional[str]:
    try:
        return cls.__module__ + "." + cls.__qualname__
    except:
        return None
