from __future__ import annotations

import logging
from typing import Callable
from unittest.mock import patch

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal
from coola.equality import EqualityConfig
from coola.equality.testers import EqualityTester

from karbonn.metric.state import (
    BaseState,
    ErrorState,
    MeanErrorState,
    is_state_config,
    setup_state,
)
from karbonn.metric.state.base import StateEqualityComparator
from karbonn.testing import objectory_available
from karbonn.utils.imports import is_objectory_available
from karbonn.utils.tracker import ScalableTensorTracker
from tests.unit.helpers import ExamplePair

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


#####################################
#     Tests for is_state_config     #
#####################################


@objectory_available
def test_is_state_config_true() -> None:
    assert is_state_config({OBJECT_TARGET: "karbonn.metric.state.ErrorState"})


@objectory_available
def test_is_state_config_false() -> None:
    assert not is_state_config({OBJECT_TARGET: "torch.device"})


#################################
#     Tests for setup_state     #
#################################


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


#############################################
#     Tests for StateEqualityComparator     #
#############################################

STATE_FUNCTIONS = [objects_are_equal, objects_are_allclose]

STATE_EQUAL = [
    pytest.param(
        ExamplePair(actual=ErrorState(), expected=ErrorState()),
        id="error state",
    ),
    pytest.param(
        ExamplePair(actual=MeanErrorState(), expected=MeanErrorState()),
        id="mean error state",
    ),
    pytest.param(
        ExamplePair(
            actual=ErrorState(
                ScalableTensorTracker(count=4, total=10.0, min_value=0.0, max_value=5.0)
            ),
            expected=ErrorState(
                ScalableTensorTracker(count=4, total=10.0, min_value=0.0, max_value=5.0)
            ),
        ),
        id="error state with elements",
    ),
]


STATE_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=ErrorState(),
            expected=MeanErrorState(),
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
    pytest.param(
        ExamplePair(
            actual=ErrorState(),
            expected=ErrorState(
                ScalableTensorTracker(count=4, total=10.0, min_value=0.0, max_value=5.0)
            ),
            expected_message="objects are not equal:",
        ),
        id="different elements",
    ),
]


def test_state_equality_comparator_repr() -> None:
    assert repr(StateEqualityComparator()) == "StateEqualityComparator()"


def test_state_equality_comparator_str() -> None:
    assert str(StateEqualityComparator()) == "StateEqualityComparator()"


def test_state_equality_comparator__eq__true() -> None:
    assert StateEqualityComparator() == StateEqualityComparator()


def test_state_equality_comparator__eq__false() -> None:
    assert StateEqualityComparator() != 123


def test_state_equality_comparator_clone() -> None:
    op = StateEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


def test_state_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    x = ErrorState()
    assert StateEqualityComparator().equal(x, x, config)


@pytest.mark.parametrize("example", STATE_EQUAL)
def test_state_equality_comparator_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = StateEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", STATE_EQUAL)
def test_state_equality_comparator_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = StateEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", STATE_NOT_EQUAL)
def test_state_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = StateEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", STATE_NOT_EQUAL)
def test_state_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = StateEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@pytest.mark.parametrize("function", STATE_FUNCTIONS)
@pytest.mark.parametrize("example", STATE_EQUAL)
@pytest.mark.parametrize("show_difference", [True, False])
def test_objects_are_equal_true(
    function: Callable,
    example: ExamplePair,
    show_difference: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert function(example.actual, example.expected, show_difference=show_difference)
        assert not caplog.messages


@pytest.mark.parametrize("function", STATE_FUNCTIONS)
@pytest.mark.parametrize("example", STATE_NOT_EQUAL)
def test_objects_are_equal_false(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.actual, example.expected)
        assert not caplog.messages


@pytest.mark.parametrize("function", STATE_FUNCTIONS)
@pytest.mark.parametrize("example", STATE_NOT_EQUAL)
def test_objects_are_equal_false_show_difference(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.actual, example.expected, show_difference=True)
        assert caplog.messages[-1].startswith(example.expected_message)
