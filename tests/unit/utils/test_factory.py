from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from torch import nn

from karbonn.testing import objectory_available
from karbonn.utils import create_sequential, is_module_config, setup_module
from karbonn.utils.imports import is_objectory_available

if TYPE_CHECKING:
    from collections.abc import Sequence

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"

######################################
#     Tests for is_module_config     #
######################################


@objectory_available
def test_is_module_config_true() -> None:
    assert is_module_config({OBJECT_TARGET: "torch.nn.Identity"})


@objectory_available
def test_is_module_config_false() -> None:
    assert not is_module_config({OBJECT_TARGET: "torch.device"})


##################################
#     Tests for setup_module     #
##################################


@objectory_available
@pytest.mark.parametrize("module", [torch.nn.ReLU(), {OBJECT_TARGET: "torch.nn.ReLU"}])
def test_setup_module(module: torch.nn.Module | dict) -> None:
    assert isinstance(setup_module(module), torch.nn.ReLU)


@objectory_available
def test_setup_module_object() -> None:
    module = torch.nn.ReLU()
    assert setup_module(module) is module


@objectory_available
def test_setup_module_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_module({OBJECT_TARGET: "torch.device", "type": "cpu"}), torch.device
        )
        assert caplog.messages


def test_setup_module_object_no_objectory() -> None:
    with (
        patch("karbonn.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match="`objectory` package is required but not installed."),
    ):
        setup_module({OBJECT_TARGET: "torch.nn.ReLU"})


#######################################
#     Tests for create_sequential     #
#######################################


@objectory_available
@pytest.mark.parametrize(
    "modules",
    [
        [{OBJECT_TARGET: "torch.nn.Linear", "in_features": 4, "out_features": 6}, nn.ReLU()],
        ({OBJECT_TARGET: "torch.nn.Linear", "in_features": 4, "out_features": 6}, nn.ReLU()),
    ],
)
def test_create_sequential(modules: Sequence) -> None:
    module = create_sequential(modules)
    assert isinstance(module, nn.Sequential)
    assert len(module) == 2
    assert isinstance(module[0], nn.Linear)
    assert isinstance(module[1], nn.ReLU)


def test_create_sequential_empty() -> None:
    module = create_sequential([])
    assert isinstance(module, nn.Sequential)
    assert len(module) == 0
