import re
from tracer.representer import NoRaiseRichRegistryRepresenter, get_full_qualified_name
from typing import Any

representer = NoRaiseRichRegistryRepresenter()


def repr(
    obj: Any,
    skip_types: set[str] = None,
):
    return representer.repr(
        obj,
        skip_types=skip_types,
    )


class SimpleClass:
    def __init__(self):
        self.val = 1


class SimpleNestClass:
    def __init__(self):
        self.nest = SimpleClass()
        self.val = 2


class SimpleReprClass:
    def __repr__(self):
        return "SimpleReprClass with val 3"


def compare(
    a: Any,
    b: str,
):
    return re.sub(
        r"0x[0-9a-f]{12}",
        r"0x000000000000",
        a.strip(),
    ) == re.sub(
        r"0x[0-9a-f]{12}",
        r"0x000000000000",
        b.strip(),
    )


def test_repr_dict():
    obj = {"a": 1}
    out = """
{'a': 1}
"""
    assert compare(repr(obj), out)


def test_repr_simple_class():
    obj = SimpleClass()
    out = """
SimpleClass at 0x7ffff65235f0
└── val: 
    └── 1
(bases: object)
"""
    assert compare(repr(obj), out)


def test_repr_simple_nest_class():
    obj = SimpleNestClass()
    out = """
SimpleNestClass at 0x7ffff65235f0
├── nest: 
│   └── val: 
│       └── 1
└── val: 
    └── 2
(bases: object)
"""
    assert compare(repr(obj), out)


def test_repr_simple_nest_class_with_skip():
    obj = SimpleNestClass()
    out = """
SimpleNestClass at 0x7ffff6523560
├── nest: 
│   └── <test_repr.SimpleClass object at 0x7ffff6523770>
└── val: 
    └── 2
(bases: object)
"""
    assert compare(
        repr(obj, skip_types={get_full_qualified_name(SimpleClass())}),
        out,
    )


def test_repr_simple_repr_class_with_skip_prefix():
    obj = SimpleNestClass()
    out = """
<test_repr.SimpleNestClass object at 0x7ffff6523770>
"""
    assert compare(
        repr(obj, skip_types={get_full_qualified_name(SimpleReprClass()).split(".")[0]}),
        out,
    )


def test_repr_dict_with_skip():
    obj = {"a": SimpleClass()}
    out = """
{'a': <test_repr.SimpleClass object at 0x7ffff65235f0>}
"""
    assert compare(
        repr(obj, skip_types={get_full_qualified_name(SimpleClass())}),
        out,
    )
