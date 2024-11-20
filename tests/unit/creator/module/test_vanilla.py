from __future__ import annotations

from torch import nn

from karbonn.creator.module import ModuleCreator
from karbonn.testing import objectory_available
from karbonn.utils.imports import is_objectory_available

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"

####################################
#     Tests for ModuleCreator     #
####################################


def test_module_creator_repr() -> None:
    assert repr(
        ModuleCreator(
            module={
                OBJECT_TARGET: "torch.nn.Linear",
                "in_features": 4,
                "out_features": 6,
            },
        )
    ).startswith("ModuleCreator")


def test_module_creator_str() -> None:
    assert str(
        ModuleCreator(
            module={
                OBJECT_TARGET: "torch.nn.Linear",
                "in_features": 4,
                "out_features": 6,
            },
        )
    ).startswith("ModuleCreator")


@objectory_available
def test_module_creator_create_dict() -> None:
    assert isinstance(
        ModuleCreator(
            module={
                OBJECT_TARGET: "torch.nn.Linear",
                "in_features": 4,
                "out_features": 6,
            },
        ).create(),
        nn.Linear,
    )


def test_module_creator_create_object() -> None:
    module = nn.Linear(in_features=4, out_features=6)
    assert ModuleCreator(module=module).create() is module
