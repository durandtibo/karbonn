from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
import torch

from karbonn.metric.state import BaseState, ErrorState, is_state_config, setup_state
from karbonn.testing import objectory_available
from karbonn.utils.imports import is_objectory_available

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"

######################################
#     Tests for is_state_config     #
######################################


@objectory_available
def test_is_state_config_true() -> None:
    assert is_state_config({OBJECT_TARGET: "karbonn.metric.state.ErrorState"})


@objectory_available
def test_is_state_config_false() -> None:
    assert not is_state_config({OBJECT_TARGET: "torch.device"})


##################################
#     Tests for setup_state     #
##################################


@objectory_available
@pytest.mark.parametrize(
    "state", [ErrorState(), {OBJECT_TARGET: "karbonn.metric.state.ErrorState"}]
)
def test_setup_state(state: BaseState | dict) -> None:
    assert isinstance(setup_state(state), ErrorState)


@objectory_available
def test_setup_state_object() -> None:
    state = ErrorState()
    assert setup_state(state) is state


@objectory_available
def test_setup_state_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_state({OBJECT_TARGET: "torch.device", "type": "cpu"}), torch.device)
        assert caplog.messages


def test_setup_state_object_no_objectory() -> None:
    with (
        patch("karbonn.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match="'objectory' package is required but not installed."),
    ):
        setup_state({OBJECT_TARGET: "karbonn.metric.state.ErrorState"})
