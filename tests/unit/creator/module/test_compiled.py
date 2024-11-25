from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from torch import nn

from karbonn.creator.module import (
    BaseModuleCreator,
    CompiledModuleCreator,
    ModuleCreator,
)
from karbonn.testing import objectory_available
from karbonn.utils.imports import is_objectory_available

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"

###########################################
#     Tests for CompiledModuleCreator     #
###########################################


def test_compiled_module_creator_repr() -> None:
    assert repr(
        CompiledModuleCreator(
            creator=ModuleCreator(
                module={
                    OBJECT_TARGET: "torch.nn.Linear",
                    "in_features": 4,
                    "out_features": 6,
                }
            )
        )
    ).startswith("CompiledModuleCreator")


def test_compiled_module_creator_str() -> None:
    assert str(
        CompiledModuleCreator(
            creator=ModuleCreator(
                module={
                    OBJECT_TARGET: "torch.nn.Linear",
                    "in_features": 4,
                    "out_features": 6,
                }
            )
        )
    ).startswith("CompiledModuleCreator")


@objectory_available
@pytest.mark.parametrize(
    "creator",
    [
        ModuleCreator(
            module={OBJECT_TARGET: "torch.nn.Linear", "in_features": 4, "out_features": 6}
        ),
        {
            OBJECT_TARGET: "karbonn.creator.module.ModuleCreator",
            "module": {OBJECT_TARGET: "torch.nn.Linear", "in_features": 4, "out_features": 6},
        },
    ],
)
def test_compiled_module_creator_create(creator: BaseModuleCreator | dict) -> None:
    module = CompiledModuleCreator(creator=creator).create()
    assert isinstance(module, nn.Module)
    assert not isinstance(module, nn.Linear)


@objectory_available
def test_compiled_module_creator_create_with_config() -> None:
    module = CompiledModuleCreator(
        creator=ModuleCreator(
            module={OBJECT_TARGET: "torch.nn.Linear", "in_features": 4, "out_features": 6}
        ),
        config={"mode": "default"},
    ).create()
    assert isinstance(module, nn.Module)
    assert not isinstance(module, nn.Linear)


def test_compiled_module_creator_create_mock() -> None:
    module = nn.Linear(4, 6)
    creator = CompiledModuleCreator(
        creator=Mock(spec=BaseModuleCreator, create=Mock(return_value=module)),
    )
    with patch("karbonn.creator.module.compiled.torch.compile") as compile_mock:
        creator.create()
        compile_mock.assert_called_once_with(module)


def test_compiled_module_creator_create_mock_with_config() -> None:
    module = nn.Linear(4, 6)
    creator = CompiledModuleCreator(
        creator=Mock(spec=BaseModuleCreator, create=Mock(return_value=module)),
        config={"mode": "default"},
    )
    with patch("karbonn.creator.module.compiled.torch.compile") as compile_mock:
        creator.create()
        compile_mock.assert_called_once_with(module, mode="default")
