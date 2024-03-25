from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
import torch

from karbonn.utils.imports import is_objectory_available

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"

from karbonn.testing import objectory_available
from karbonn.utils import is_module_config, setup_module

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
