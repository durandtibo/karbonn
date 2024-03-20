from __future__ import annotations

import logging

import pytest
import torch
from objectory import OBJECT_TARGET

from karbonn import is_module_config, setup_module

######################################
#     Tests for is_module_config     #
######################################


def test_is_module_config_true() -> None:
    assert is_module_config({OBJECT_TARGET: "torch.nn.Identity"})


def test_is_module_config_false() -> None:
    assert not is_module_config({OBJECT_TARGET: "torch.device"})


##################################
#     Tests for setup_module     #
##################################


@pytest.mark.parametrize("module", [torch.nn.ReLU(), {OBJECT_TARGET: "torch.nn.ReLU"}])
def test_setup_module(module: torch.nn.Module | dict) -> None:
    assert isinstance(setup_module(module), torch.nn.ReLU)


def test_setup_module_object() -> None:
    module = torch.nn.ReLU()
    assert setup_module(module) is module


def test_setup_module_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_module({OBJECT_TARGET: "torch.device", "type": "cpu"}), torch.device
        )
        assert caplog.messages
