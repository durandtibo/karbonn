from __future__ import annotations

from typing import Any

import pytest
import torch.nn
from torch import nn

from karbonn.creator import Creator
from karbonn.testing import objectory_available
from karbonn.utils.imports import is_objectory_available

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"

OBJECTS_OR_CONFIGS = [
    torch.nn.Linear(in_features=4, out_features=6),
    {OBJECT_TARGET: "torch.nn.Linear", "in_features": 4, "out_features": 6},
]

#############################
#     Tests for Creator     #
#############################


@pytest.mark.parametrize("obj_or_config", OBJECTS_OR_CONFIGS)
def test_creator_repr(obj_or_config: Any) -> None:
    assert repr(Creator(obj_or_config=obj_or_config)).startswith("Creator")


@pytest.mark.parametrize("obj_or_config", OBJECTS_OR_CONFIGS)
def test_creator_str(obj_or_config: Any) -> None:
    assert str(Creator(obj_or_config=obj_or_config)).startswith("Creator")


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
