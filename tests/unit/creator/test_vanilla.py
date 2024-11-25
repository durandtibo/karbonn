from __future__ import annotations

from torch import nn

from karbonn.creator import Creator
from karbonn.testing import objectory_available
from karbonn.utils.imports import is_objectory_available

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"

#############################
#     Tests for Creator     #
#############################


def test_creator_repr() -> None:
    assert repr(
        Creator(
            obj_or_config={
                OBJECT_TARGET: "torch.nn.Linear",
                "in_features": 4,
                "out_features": 6,
            },
        )
    ).startswith("Creator")


def test_creator_str() -> None:
    assert str(
        Creator(
            obj_or_config={
                OBJECT_TARGET: "torch.nn.Linear",
                "in_features": 4,
                "out_features": 6,
            },
        )
    ).startswith("Creator")


@objectory_available
def test_creator_create_dict() -> None:
    assert isinstance(
        Creator(
            obj_or_config={
                OBJECT_TARGET: "torch.nn.Linear",
                "in_features": 4,
                "out_features": 6,
            },
        ).create(),
        nn.Linear,
    )


def test_creator_create_object() -> None:
    obj = nn.Linear(in_features=4, out_features=6)
    assert Creator(obj_or_config=obj).create() is obj
