from __future__ import annotations

import pytest
import torch
from minrecord import MinScalarRecord

from karbonn.metric import EmptyMetricError
from karbonn.metric.state import ErrorState, MeanErrorState

################################
#     Tests for ErrorState     #
################################


def test_error_state_repr() -> None:
    assert repr(ErrorState()).startswith("ErrorState(")


def test_error_state_str() -> None:
    assert str(ErrorState()).startswith("ErrorState(")


def test_error_state_get_records() -> None:
    records = ErrorState().get_records()
    assert len(records) == 4
    assert isinstance(records[0], MinScalarRecord)
    assert records[0].name == "mean"
    assert isinstance(records[1], MinScalarRecord)
    assert records[1].name == "min"
    assert isinstance(records[2], MinScalarRecord)
    assert records[2].name == "max"
    assert isinstance(records[3], MinScalarRecord)
    assert records[3].name == "sum"


@pytest.mark.parametrize("prefix", ["", "prefix_"])
@pytest.mark.parametrize("suffix", ["", "_suffix"])
def test_error_state_get_records_prefix_suffix(prefix: str, suffix: str) -> None:
    records = ErrorState().get_records(prefix, suffix)
    assert len(records) == 4
    assert isinstance(records[0], MinScalarRecord)
    assert records[0].name == f"{prefix}mean{suffix}"
    assert isinstance(records[1], MinScalarRecord)
    assert records[1].name == f"{prefix}min{suffix}"
    assert isinstance(records[2], MinScalarRecord)
    assert records[2].name == f"{prefix}max{suffix}"
    assert isinstance(records[3], MinScalarRecord)
    assert records[3].name == f"{prefix}sum{suffix}"


def test_error_state_reset() -> None:
    state = ErrorState()
    state.update(torch.arange(6))
    assert state.num_predictions == 6
    state.reset()
    assert state.num_predictions == 0


def test_error_state_update_1d() -> None:
    state = ErrorState()
    state.update(torch.arange(6))
    assert state._meter.count == 6
    assert state._meter.sum() == 15.0
    assert state._meter.max() == 5.0
    assert state._meter.min() == 0.0


def test_error_state_update_2d() -> None:
    state = ErrorState()
    state.update(torch.arange(6).view(2, 3))
    assert state._meter.count == 6
    assert state._meter.sum() == 15.0
    assert state._meter.max() == 5.0
    assert state._meter.min() == 0.0


def test_error_state_value() -> None:
    state = ErrorState()
    state.update(torch.arange(6))
    assert state.value() == {
        "mean": 2.5,
        "min": 0.0,
        "max": 5.0,
        "sum": 15.0,
        "num_predictions": 6,
    }


def test_error_state_value_num_predictions_false() -> None:
    state = ErrorState(track_num_predictions=False)
    state.update(torch.arange(6))
    assert state.value() == {"mean": 2.5, "min": 0.0, "max": 5.0, "sum": 15.0}


@pytest.mark.parametrize("prefix", ["", "prefix_"])
@pytest.mark.parametrize("suffix", ["", "_suffix"])
def test_error_state_value_prefix_suffix(prefix: str, suffix: str) -> None:
    state = ErrorState()
    state.update(torch.arange(6))
    assert state.value(prefix, suffix) == {
        f"{prefix}mean{suffix}": 2.5,
        f"{prefix}min{suffix}": 0.0,
        f"{prefix}max{suffix}": 5.0,
        f"{prefix}sum{suffix}": 15.0,
        f"{prefix}num_predictions{suffix}": 6,
    }


def test_error_state_value_empty() -> None:
    state = ErrorState()
    with pytest.raises(EmptyMetricError, match="ErrorState is empty"):
        state.value()


####################################
#     Tests for MeanErrorState     #
####################################


def test_mean_error_state_repr() -> None:
    assert repr(MeanErrorState()).startswith("MeanErrorState(")


def test_mean_error_state_str() -> None:
    assert str(MeanErrorState()).startswith("MeanErrorState(")


def test_mean_error_state_get_records() -> None:
    records = MeanErrorState().get_records()
    assert len(records) == 1
    assert isinstance(records[0], MinScalarRecord)
    assert records[0].name == "mean"


@pytest.mark.parametrize("prefix", ["", "prefix_"])
@pytest.mark.parametrize("suffix", ["", "_suffix"])
def test_mean_error_state_get_records_prefix_suffix(prefix: str, suffix: str) -> None:
    records = MeanErrorState().get_records(prefix, suffix)
    assert len(records) == 1
    assert isinstance(records[0], MinScalarRecord)
    assert records[0].name == f"{prefix}mean{suffix}"


def test_mean_error_state_reset() -> None:
    state = MeanErrorState()
    state.update(torch.arange(6))
    assert state.num_predictions == 6
    state.reset()
    assert state.num_predictions == 0


def test_mean_error_state_update_1d() -> None:
    state = MeanErrorState()
    state.update(torch.arange(6))
    assert state._meter.count == 6
    assert state._meter.sum() == 15.0


def test_mean_error_state_update_2d() -> None:
    state = MeanErrorState()
    state.update(torch.arange(6).view(2, 3))
    assert state._meter.count == 6
    assert state._meter.sum() == 15.0


def test_mean_error_state_value() -> None:
    state = MeanErrorState()
    state.update(torch.arange(6))
    assert state.value() == {"mean": 2.5, "num_predictions": 6}


def test_mean_error_state_value_correct() -> None:
    state = MeanErrorState()
    state.update(torch.zeros(4))
    assert state.value() == {"mean": 0.0, "num_predictions": 4}


def test_mean_error_state_value_track_num_predictions_false() -> None:
    state = MeanErrorState(track_num_predictions=False)
    state.update(torch.arange(6))
    assert state.value() == {"mean": 2.5}


@pytest.mark.parametrize("prefix", ["", "prefix_"])
@pytest.mark.parametrize("suffix", ["", "_suffix"])
def test_mean_error_state_value_prefix_suffix(prefix: str, suffix: str) -> None:
    state = MeanErrorState()
    state.update(torch.arange(6))
    assert state.value(prefix, suffix) == {
        f"{prefix}mean{suffix}": 2.5,
        f"{prefix}num_predictions{suffix}": 6,
    }


def test_mean_error_state_value_empty() -> None:
    state = MeanErrorState()
    with pytest.raises(EmptyMetricError, match="MeanErrorState is empty"):
        state.value()
