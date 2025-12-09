from typing import Any

from tracer.deep_copy import deepcopy


def deep_copy(
    v: Any,
) -> Any:
    return deepcopy(v)


def deep_equal(
    v1: Any,
    v2: Any,
) -> bool:

    if v1 is v2:
        return True

    if type(v1) != type(v2):
        return False

    cls = type(v1)

    if cls.__eq__ is not object.__eq__:
        try:
            return v1 == v2
        except:
            return False

    if hasattr(v1, "__dict__"):
        d1 = vars(v1)
        d2 = vars(v2)

        if d1.keys() != d2.keys():
            return False

        for key in d1.keys():
            if not deep_equal(d1[key], d2[key]):
                return False
        return True

    if hasattr(v1, "__slots__"):
        slots = cls.__slots__
        if isinstance(slots, str):
            slots = (slots,)

        for slot in slots:
            try:
                if not deep_equal(getattr(v1, slot), getattr(v2, slot)):
                    return False
            except AttributeError:
                return False
        return True

    return False
