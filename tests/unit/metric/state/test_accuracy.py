from __future__ import annotations

import pytest
import torch
from minrecord import MaxScalarRecord

from karbonn.metric import EmptyMetricError
from karbonn.metric.state import AccuracyState

###################################
#     Tests for AccuracyState     #
###################################


def test_accuracy_state_repr() -> None:
    assert repr(AccuracyState()).startswith("AccuracyState(")


def test_accuracy_state_str() -> None:
    assert str(AccuracyState()).startswith("AccuracyState(")


def test_accuracy_state_get_records() -> None:
    records = AccuracyState().get_records()
    assert len(records) == 1
    assert isinstance(records[0], MaxScalarRecord)
    assert records[0].name == "accuracy"


@pytest.mark.parametrize("prefix", ["", "prefix_"])
@pytest.mark.parametrize("suffix", ["", "_suffix"])
def test_accuracy_state_get_records_prefix_suffix(prefix: str, suffix: str) -> None:
    records = AccuracyState().get_records(prefix, suffix)
    assert len(records) == 1
    assert isinstance(records[0], MaxScalarRecord)
    assert records[0].name == f"{prefix}accuracy{suffix}"


def test_accuracy_state_reset() -> None:
    state = AccuracyState()
    state.update(torch.eye(2))
    assert state.num_predictions == 4
    state.reset()
    assert state.num_predictions == 0


def test_accuracy_state_update_1d() -> None:
    state = AccuracyState()
    state.update(torch.ones(4))
    assert state._tracker.count == 4
    assert state._tracker.sum() == 4.0


def test_accuracy_state_update_2d() -> None:
    state = AccuracyState()
    state.update(torch.ones(2, 3))
    assert state._tracker.count == 6
    assert state._tracker.sum() == 6.0


def test_accuracy_state_value_correct() -> None:
    state = AccuracyState()
    state.update(torch.ones(4))
    assert state.value() == {"accuracy": 1.0, "num_predictions": 4}


def test_accuracy_state_value_partially_correct() -> None:
    state = AccuracyState()
    state.update(torch.eye(2))
    assert state.value() == {"accuracy": 0.5, "num_predictions": 4}


def test_accuracy_state_value_incorrect() -> None:
    state = AccuracyState()
    state.update(torch.zeros(4))
    assert state.value() == {"accuracy": 0.0, "num_predictions": 4}


def test_accuracy_state_value_track_num_predictions_false() -> None:
    state = AccuracyState(track_num_predictions=False)
    state.update(torch.eye(2))
    assert state.value() == {"accuracy": 0.5}


@pytest.mark.parametrize("prefix", ["", "prefix_"])
@pytest.mark.parametrize("suffix", ["", "_suffix"])
def test_accuracy_state_value_prefix_suffix(prefix: str, suffix: str) -> None:
    state = AccuracyState()
    state.update(torch.eye(2))
    assert state.value(prefix, suffix) == {
        f"{prefix}accuracy{suffix}": 0.5,
        f"{prefix}num_predictions{suffix}": 4,
    }


def test_accuracy_state_value_empty() -> None:
    state = AccuracyState()
    with pytest.raises(EmptyMetricError, match="AccuracyState is empty"):
        state.value()
