from __future__ import annotations

import pytest
import torch
from minrecord import MinScalarRecord

from karbonn.metric import EmptyMetricError
from karbonn.metric.state import MeanErrorState

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
