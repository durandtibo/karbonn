from __future__ import annotations

import torch.nn
from torch import nn

from karbonn.creator import CreatorList
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
