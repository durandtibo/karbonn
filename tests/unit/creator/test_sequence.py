from __future__ import annotations

import torch.nn
from torch import nn

from karbonn.creator import Creator, CreatorList, CreatorTuple, ListCreator
from karbonn.testing import objectory_available
from karbonn.utils.imports import is_objectory_available

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"

#################################
#     Tests for CreatorList     #
#################################


def test_creator_list_repr() -> None:
    assert repr(
        CreatorList(
            items=[
                {
                    OBJECT_TARGET: "torch.nn.Linear",
                    "in_features": 4,
                    "out_features": 6,
                },
                torch.nn.Identity(),
            ],
        )
    ).startswith("CreatorList")


def test_creator_list_str() -> None:
    assert str(
        CreatorList(
            items=[
                {
                    OBJECT_TARGET: "torch.nn.Linear",
                    "in_features": 4,
                    "out_features": 6,
                },
                torch.nn.Identity(),
            ],
        )
    ).startswith("CreatorList")


@objectory_available
def test_creator_list_create_one_item() -> None:
    obj = CreatorList(
        items=[
            {
                OBJECT_TARGET: "torch.nn.Linear",
                "in_features": 4,
                "out_features": 6,
            }
        ],
    ).create()
    assert isinstance(obj, list)
    assert len(obj) == 1
    assert isinstance(obj[0], nn.Linear)


@objectory_available
def test_creator_list_create_two_items() -> None:
    obj = CreatorList(
        items=[
            {
                OBJECT_TARGET: "torch.nn.Linear",
                "in_features": 4,
                "out_features": 6,
            },
            torch.nn.Identity(),
        ],
    ).create()
    assert isinstance(obj, list)
    assert len(obj) == 2
    assert isinstance(obj[0], nn.Linear)
    assert isinstance(obj[1], nn.Identity)


##################################
#     Tests for CreatorTuple     #
##################################


def test_creator_tuple_repr() -> None:
    assert repr(
        CreatorTuple(
            items=[
                {
                    OBJECT_TARGET: "torch.nn.Linear",
                    "in_features": 4,
                    "out_features": 6,
                },
                torch.nn.Identity(),
            ],
        )
    ).startswith("CreatorTuple")


def test_creator_tuple_str() -> None:
    assert str(
        CreatorTuple(
            items=[
                {
                    OBJECT_TARGET: "torch.nn.Linear",
                    "in_features": 4,
                    "out_features": 6,
                },
                torch.nn.Identity(),
            ],
        )
    ).startswith("CreatorTuple")


@objectory_available
def test_creator_tuple_create_one_item() -> None:
    obj = CreatorTuple(
        items=[
            {
                OBJECT_TARGET: "torch.nn.Linear",
                "in_features": 4,
                "out_features": 6,
            }
        ],
    ).create()
    assert isinstance(obj, tuple)
    assert len(obj) == 1
    assert isinstance(obj[0], nn.Linear)


@objectory_available
def test_creator_tuple_create_two_items() -> None:
    obj = CreatorTuple(
        items=[
            {
                OBJECT_TARGET: "torch.nn.Linear",
                "in_features": 4,
                "out_features": 6,
            },
            torch.nn.Identity(),
        ],
    ).create()
    assert isinstance(obj, tuple)
    assert len(obj) == 2
    assert isinstance(obj[0], nn.Linear)
    assert isinstance(obj[1], nn.Identity)


#################################
#     Tests for ListCreator     #
#################################


def test_list_creator_repr() -> None:
    assert repr(
        ListCreator(
            creators=[
                {
                    OBJECT_TARGET: "karbonn.creator.Creator",
                    "obj_or_config": torch.nn.Linear(in_features=4, out_features=6),
                },
                Creator(torch.nn.Identity()),
            ],
        )
    ).startswith("ListCreator")


def test_list_creator_str() -> None:
    assert str(
        ListCreator(
            creators=[
                {
                    OBJECT_TARGET: "karbonn.creator.Creator",
                    "obj_or_config": torch.nn.Linear(in_features=4, out_features=6),
                },
                Creator(torch.nn.Identity()),
            ],
        )
    ).startswith("ListCreator")


@objectory_available
def test_list_creator_create_one_item() -> None:
    obj = ListCreator(
        creators=[
            {
                OBJECT_TARGET: "karbonn.creator.Creator",
                "obj_or_config": torch.nn.Linear(in_features=4, out_features=6),
            },
        ],
    ).create()
    assert isinstance(obj, list)
    assert len(obj) == 1
    assert isinstance(obj[0], nn.Linear)


@objectory_available
def test_list_creator_create_two_creators() -> None:
    obj = ListCreator(
        creators=[
            {
                OBJECT_TARGET: "karbonn.creator.Creator",
                "obj_or_config": torch.nn.Linear(in_features=4, out_features=6),
            },
            Creator(torch.nn.Identity()),
        ],
    ).create()
    assert isinstance(obj, list)
    assert len(obj) == 2
    assert isinstance(obj[0], nn.Linear)
    assert isinstance(obj[1], nn.Identity)
