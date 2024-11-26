from __future__ import annotations

import pytest
from torch import nn

from karbonn.creator import BaseCreator, CacheCreator, Creator
from karbonn.testing import objectory_available
from karbonn.utils.imports import is_objectory_available

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"

CREATORS_OR_CONFIGS = [
    {
        OBJECT_TARGET: "karbonn.creator.Creator",
        "obj_or_config": {OBJECT_TARGET: "torch.nn.Linear", "in_features": 4, "out_features": 6},
    },
    Creator({OBJECT_TARGET: "torch.nn.Linear", "in_features": 4, "out_features": 6}),
]

##################################
#     Tests for CacheCreator     #
##################################


@objectory_available
@pytest.mark.parametrize("creator", CREATORS_OR_CONFIGS)
def test_creator_repr(creator: BaseCreator | dict) -> None:
    assert repr(CacheCreator(creator=creator)).startswith("CacheCreator")


@objectory_available
@pytest.mark.parametrize("creator", CREATORS_OR_CONFIGS)
def test_creator_str(creator: BaseCreator | dict) -> None:
    assert str(CacheCreator(creator=creator)).startswith("CacheCreator")


@objectory_available
@pytest.mark.parametrize("creator", CREATORS_OR_CONFIGS)
def test_creator_create(creator: BaseCreator | dict) -> None:
    assert isinstance(CacheCreator(creator).create(), nn.Linear)


def test_creator_create_copy_false() -> None:
    obj = nn.Linear(in_features=4, out_features=6)
    creator = CacheCreator(creator=Creator(obj), copy=False)
    obj1 = creator.create()
    obj2 = creator.create()
    assert obj is obj1
    assert obj is obj2
    assert obj1 is obj2
    assert isinstance(obj1, nn.Linear)
    assert isinstance(obj2, nn.Linear)


def test_creator_create_copy_true() -> None:
    obj = nn.Linear(in_features=4, out_features=6)
    creator = CacheCreator(creator=Creator(obj))
    obj1 = creator.create()
    obj2 = creator.create()
    assert obj is not obj1
    assert obj is not obj2
    assert obj1 is not obj2
    assert isinstance(obj1, nn.Linear)
    assert isinstance(obj2, nn.Linear)
