from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal
from minrecord import MaxScalarRecord, MinScalarRecord

from karbonn.metric import EmptyMetricError
from karbonn.metric.state import AccuracyState, ExtendedAccuracyState

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
    assert objects_are_equal(state.value(), {"accuracy": 1.0, "num_predictions": 4})


def test_accuracy_state_value_partially_correct() -> None:
    state = AccuracyState()
    state.update(torch.eye(2))
    assert objects_are_equal(state.value(), {"accuracy": 0.5, "num_predictions": 4})


def test_accuracy_state_value_incorrect() -> None:
    state = AccuracyState()
    state.update(torch.zeros(4))
    assert objects_are_equal(state.value(), {"accuracy": 0.0, "num_predictions": 4})


def test_accuracy_state_value_track_num_predictions_false() -> None:
    state = AccuracyState(track_num_predictions=False)
    state.update(torch.eye(2))
    assert state.value() == {"accuracy": 0.5}


@pytest.mark.parametrize("prefix", ["", "prefix_"])
@pytest.mark.parametrize("suffix", ["", "_suffix"])
def test_accuracy_state_value_prefix_suffix(prefix: str, suffix: str) -> None:
    state = AccuracyState()
    state.update(torch.eye(2))
    assert objects_are_equal(
        state.value(prefix, suffix),
        {
            f"{prefix}accuracy{suffix}": 0.5,
            f"{prefix}num_predictions{suffix}": 4,
        },
    )


def test_accuracy_state_value_empty() -> None:
    state = AccuracyState()
    with pytest.raises(EmptyMetricError, match="AccuracyState is empty"):
        state.value()


###########################################
#     Tests for ExtendedAccuracyState     #
###########################################


def test_extended_accuracy_state_repr() -> None:
    assert repr(ExtendedAccuracyState()).startswith("ExtendedAccuracyState(")


def test_extended_accuracy_state_str() -> None:
    assert str(ExtendedAccuracyState()).startswith("ExtendedAccuracyState(")


def test_extended_accuracy_state_get_records() -> None:
    records = ExtendedAccuracyState().get_records()
    assert len(records) == 4
    assert isinstance(records[0], MaxScalarRecord)
    assert records[0].name == "accuracy"
    assert isinstance(records[1], MinScalarRecord)
    assert records[1].name == "error"
    assert isinstance(records[2], MaxScalarRecord)
    assert records[2].name == "num_correct_predictions"
    assert isinstance(records[3], MinScalarRecord)
    assert records[3].name == "num_incorrect_predictions"


@pytest.mark.parametrize("prefix", ["", "prefix_"])
@pytest.mark.parametrize("suffix", ["", "_suffix"])
def test_extended_accuracy_state_get_records_prefix_suffix(prefix: str, suffix: str) -> None:
    records = ExtendedAccuracyState().get_records(prefix, suffix)
    assert len(records) == 4
    assert isinstance(records[0], MaxScalarRecord)
    assert records[0].name == f"{prefix}accuracy{suffix}"
    assert isinstance(records[1], MinScalarRecord)
    assert records[1].name == f"{prefix}error{suffix}"
    assert isinstance(records[2], MaxScalarRecord)
    assert records[2].name == f"{prefix}num_correct_predictions{suffix}"
    assert isinstance(records[3], MinScalarRecord)
    assert records[3].name == f"{prefix}num_incorrect_predictions{suffix}"


def test_extended_accuracy_state_reset() -> None:
    state = ExtendedAccuracyState()
    state.update(torch.eye(2))
    assert state.num_predictions == 4
    state.reset()
    assert state.num_predictions == 0


def test_extended_accuracy_state_update_1d() -> None:
    state = ExtendedAccuracyState()
    state.update(torch.ones(4))
    assert state._tracker.count == 4
    assert state._tracker.sum() == 4.0


def test_extended_accuracy_state_update_2d() -> None:
    state = ExtendedAccuracyState()
    state.update(torch.ones(2, 3))
    assert state._tracker.count == 6
    assert state._tracker.sum() == 6.0


def test_extended_accuracy_state_value_correct() -> None:
    state = ExtendedAccuracyState()
    state.update(torch.ones(2, 3))
    assert objects_are_equal(
        state.value(),
        {
            "accuracy": 1.0,
            "error": 0.0,
            "num_correct_predictions": 6,
            "num_incorrect_predictions": 0,
            "num_predictions": 6,
        },
    )


def test_extended_accuracy_state_value_partially_correct() -> None:
    state = ExtendedAccuracyState()
    state.update(torch.eye(2))
    assert objects_are_equal(
        state.value(),
        {
            "accuracy": 0.5,
            "error": 0.5,
            "num_correct_predictions": 2,
            "num_incorrect_predictions": 2,
            "num_predictions": 4,
        },
    )


def test_extended_accuracy_state_value_incorrect() -> None:
    state = ExtendedAccuracyState()
    state.update(torch.zeros(2, 3))
    assert objects_are_equal(
        state.value(),
        {
            "accuracy": 0.0,
            "error": 1.0,
            "num_correct_predictions": 0,
            "num_incorrect_predictions": 6,
            "num_predictions": 6,
        },
    )


@pytest.mark.parametrize("prefix", ["", "prefix_"])
@pytest.mark.parametrize("suffix", ["", "_suffix"])
def test_extended_accuracy_state_value_prefix_suffix(prefix: str, suffix: str) -> None:
    state = ExtendedAccuracyState()
    state.update(torch.eye(2))
    assert objects_are_equal(
        state.value(prefix, suffix),
        {
            f"{prefix}accuracy{suffix}": 0.5,
            f"{prefix}error{suffix}": 0.5,
            f"{prefix}num_correct_predictions{suffix}": 2,
            f"{prefix}num_incorrect_predictions{suffix}": 2,
            f"{prefix}num_predictions{suffix}": 4,
        },
    )


def test_extended_accuracy_state_value_track_num_predictions_false() -> None:
    state = ExtendedAccuracyState(track_num_predictions=False)
    state.update(torch.eye(2))
    assert objects_are_equal(state.value(), {"accuracy": 0.5, "error": 0.5})


def test_extended_accuracy_state_value_empty() -> None:
    state = ExtendedAccuracyState()
    with pytest.raises(EmptyMetricError, match="ExtendedAccuracyState is empty"):
        state.value()
