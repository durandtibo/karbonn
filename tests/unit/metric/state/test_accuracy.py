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


def test_accuracy_state_clone() -> None:
    state = AccuracyState(MeanTensorTracker(count=4, total=10.0))
    cloned = state.clone()
    assert state is not cloned
    assert state.equal(cloned)
    state.update(torch.ones(4))
    assert not state.equal(cloned)


def test_accuracy_state_clone_empty() -> None:
    state = AccuracyState()
    cloned = state.clone()
    assert state is not cloned
    assert state.equal(cloned)
    state.update(torch.ones(4))
    assert not state.equal(cloned)


def test_accuracy_state_equal_true() -> None:
    assert AccuracyState(MeanTensorTracker(count=4, total=10.0)).equal(
        AccuracyState(MeanTensorTracker(count=4, total=10.0))
    )


def test_accuracy_state_equal_true_empty() -> None:
    assert AccuracyState().equal(AccuracyState())


def test_accuracy_state_equal_false_different_tracker() -> None:
    assert not AccuracyState(MeanTensorTracker(count=4, total=10.0)).equal(AccuracyState())


def test_accuracy_state_equal_false_different_track_count() -> None:
    assert not AccuracyState().equal(AccuracyState(track_count=False))


def test_accuracy_state_equal_false_different_type() -> None:
    assert not AccuracyState().equal(MeanTensorTracker(count=4, total=10.0))


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
    assert state.count == 4
    state.reset()
    assert state.count == 0


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
    assert objects_are_equal(state.value(), {"accuracy": 1.0, "count": 4})


def test_accuracy_state_value_partially_correct() -> None:
    state = AccuracyState()
    state.update(torch.eye(2))
    assert objects_are_equal(state.value(), {"accuracy": 0.5, "count": 4})


def test_accuracy_state_value_incorrect() -> None:
    state = AccuracyState()
    state.update(torch.zeros(4))
    assert objects_are_equal(state.value(), {"accuracy": 0.0, "count": 4})


def test_accuracy_state_value_track_count_false() -> None:
    state = AccuracyState(track_count=False)
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
            f"{prefix}count{suffix}": 4,
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


def test_extended_accuracy_state_clone() -> None:
    state = ExtendedAccuracyState(MeanTensorTracker(count=4, total=10.0))
    cloned = state.clone()
    assert state is not cloned
    assert state.equal(cloned)
    state.update(torch.ones(4))
    assert not state.equal(cloned)


def test_extended_accuracy_state_clone_empty() -> None:
    state = ExtendedAccuracyState()
    cloned = state.clone()
    assert state is not cloned
    assert state.equal(cloned)
    state.update(torch.ones(4))
    assert not state.equal(cloned)


def test_extended_accuracy_state_equal_true() -> None:
    assert ExtendedAccuracyState(MeanTensorTracker(count=4, total=10.0)).equal(
        ExtendedAccuracyState(MeanTensorTracker(count=4, total=10.0))
    )


def test_extended_accuracy_state_equal_true_empty() -> None:
    assert ExtendedAccuracyState().equal(ExtendedAccuracyState())


def test_extended_accuracy_state_equal_false_different_tracker() -> None:
    assert not ExtendedAccuracyState(MeanTensorTracker(count=4, total=10.0)).equal(
        ExtendedAccuracyState()
    )


def test_extended_accuracy_state_equal_false_different_track_count() -> None:
    assert not ExtendedAccuracyState().equal(ExtendedAccuracyState(track_count=False))


def test_extended_accuracy_state_equal_false_different_type() -> None:
    assert not ExtendedAccuracyState().equal(MeanTensorTracker(count=4, total=10.0))


def test_extended_accuracy_state_get_records() -> None:
    assert objects_are_equal(
        ExtendedAccuracyState().get_records(),
        (
            MaxScalarRecord(name="accuracy"),
            MinScalarRecord(name="error"),
            MaxScalarRecord(name="count_correct"),
            MinScalarRecord(name="count_incorrect"),
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
            MaxScalarRecord(name=f"{prefix}count_correct{suffix}"),
            MinScalarRecord(name=f"{prefix}count_incorrect{suffix}"),
        ),
    )


def test_extended_accuracy_state_reset() -> None:
    state = ExtendedAccuracyState()
    state.update(torch.eye(2))
    assert state.count == 4
    state.reset()
    assert state.count == 0


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
            "count_correct": 6,
            "count_incorrect": 0,
            "count": 6,
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
            "count_correct": 2,
            "count_incorrect": 2,
            "count": 4,
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
            "count_correct": 0,
            "count_incorrect": 6,
            "count": 6,
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
            f"{prefix}count_correct{suffix}": 2,
            f"{prefix}count_incorrect{suffix}": 2,
            f"{prefix}count{suffix}": 4,
        },
    )


def test_extended_accuracy_state_value_track_count_false() -> None:
    state = ExtendedAccuracyState(track_count=False)
    state.update(torch.eye(2))
    assert objects_are_equal(state.value(), {"accuracy": 0.5, "error": 0.5})


def test_extended_accuracy_state_value_empty() -> None:
    state = ExtendedAccuracyState()
    with pytest.raises(EmptyMetricError, match="ExtendedAccuracyState is empty"):
        state.value()
