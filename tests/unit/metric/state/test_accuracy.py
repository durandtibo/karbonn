from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal
from minrecord import MaxScalarRecord, MinScalarRecord

from karbonn.metric import EmptyMetricError
from karbonn.metric.state import AccuracyState, ExtendedAccuracyState
from karbonn.utils.tracker import MeanTensorTracker

###################################
#     Tests for AccuracyState     #
###################################


def test_accuracy_state_repr() -> None:
    assert repr(AccuracyState()).startswith("AccuracyState(")


def test_accuracy_state_str() -> None:
    assert str(AccuracyState()).startswith("AccuracyState(")


def test_accuracy_state_get_records() -> None:
    assert objects_are_equal(
        AccuracyState().get_records(),
        (MaxScalarRecord(name="accuracy"),),
    )


@pytest.mark.parametrize("prefix", ["", "prefix_"])
@pytest.mark.parametrize("suffix", ["", "_suffix"])
def test_accuracy_state_get_records_prefix_suffix(prefix: str, suffix: str) -> None:
    assert objects_are_equal(
        AccuracyState().get_records(prefix, suffix),
        (MaxScalarRecord(name=f"{prefix}accuracy{suffix}"),),
    )


def test_accuracy_state_reset() -> None:
    state = AccuracyState()
    state.update(torch.eye(2))
    assert state.num_predictions == 4
    state.reset()
    assert state.num_predictions == 0


def test_accuracy_state_update_1d() -> None:
    state = AccuracyState()
    state.update(torch.ones(4))
    assert state._tracker.equal(MeanTensorTracker(count=4, total=4.0))


def test_accuracy_state_update_2d() -> None:
    state = AccuracyState()
    state.update(torch.ones(2, 3))
    assert state._tracker.equal(MeanTensorTracker(count=6, total=6.0))


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
    assert objects_are_equal(state.value(), {"accuracy": 0.5})


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
    assert objects_are_equal(
        ExtendedAccuracyState().get_records(),
        (
            MaxScalarRecord(name="accuracy"),
            MinScalarRecord(name="error"),
            MaxScalarRecord(name="num_correct_predictions"),
            MinScalarRecord(name="num_incorrect_predictions"),
        ),
    )


@pytest.mark.parametrize("prefix", ["", "prefix_"])
@pytest.mark.parametrize("suffix", ["", "_suffix"])
def test_extended_accuracy_state_get_records_prefix_suffix(prefix: str, suffix: str) -> None:
    assert objects_are_equal(
        ExtendedAccuracyState().get_records(prefix, suffix),
        (
            MaxScalarRecord(name=f"{prefix}accuracy{suffix}"),
            MinScalarRecord(name=f"{prefix}error{suffix}"),
            MaxScalarRecord(name=f"{prefix}num_correct_predictions{suffix}"),
            MinScalarRecord(name=f"{prefix}num_incorrect_predictions{suffix}"),
        ),
    )


def test_extended_accuracy_state_reset() -> None:
    state = ExtendedAccuracyState()
    state.update(torch.eye(2))
    assert state.num_predictions == 4
    state.reset()
    assert state.num_predictions == 0


def test_extended_accuracy_state_update_1d() -> None:
    state = ExtendedAccuracyState()
    state.update(torch.ones(4))
    assert state._tracker.equal(MeanTensorTracker(count=4, total=4.0))


def test_extended_accuracy_state_update_2d() -> None:
    state = ExtendedAccuracyState()
    state.update(torch.ones(2, 3))
    assert state._tracker.equal(MeanTensorTracker(count=6, total=6.0))


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
