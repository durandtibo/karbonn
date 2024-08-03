from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from coola import objects_are_allclose
from minrecord import MinScalarRecord

from karbonn.metric import EmptyMetricError
from karbonn.metric.state import ErrorState, ExtendedErrorState, MeanErrorState

if TYPE_CHECKING:
    from collections.abc import Sequence

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


########################################
#     Tests for ExtendedErrorState     #
########################################


def test_extended_error_state_repr() -> None:
    assert repr(ExtendedErrorState()).startswith("ExtendedErrorState(")


def test_extended_error_state_str() -> None:
    assert str(ExtendedErrorState()).startswith("ExtendedErrorState(")


@pytest.mark.parametrize("quantiles", [torch.tensor([0.5, 0.9]), [0.5, 0.9], (0.5, 0.9)])
def test_extended_error_state_init_quantiles(quantiles: torch.Tensor | Sequence[float]) -> None:
    assert ExtendedErrorState(quantiles)._quantiles.equal(
        torch.tensor([0.5, 0.9], dtype=torch.float)
    )


def test_extended_error_state_init_quantiles_empty() -> None:
    assert ExtendedErrorState()._quantiles.equal(torch.tensor([]))


def test_extended_error_state_get_records_no_quantile() -> None:
    histories = ExtendedErrorState().get_records()
    assert len(histories) == 5
    assert isinstance(histories[0], MinScalarRecord)
    assert histories[0].name == "mean"
    assert isinstance(histories[1], MinScalarRecord)
    assert histories[1].name == "median"
    assert isinstance(histories[2], MinScalarRecord)
    assert histories[2].name == "min"
    assert isinstance(histories[3], MinScalarRecord)
    assert histories[3].name == "max"
    assert isinstance(histories[4], MinScalarRecord)
    assert histories[4].name == "sum"


def test_extended_error_state_get_records_quantiles() -> None:
    histories = ExtendedErrorState(quantiles=[0.5, 0.9]).get_records()
    assert len(histories) == 7
    assert isinstance(histories[0], MinScalarRecord)
    assert histories[0].name == "mean"
    assert isinstance(histories[1], MinScalarRecord)
    assert histories[1].name == "median"
    assert isinstance(histories[2], MinScalarRecord)
    assert histories[2].name == "min"
    assert isinstance(histories[3], MinScalarRecord)
    assert histories[3].name == "max"
    assert isinstance(histories[4], MinScalarRecord)
    assert histories[4].name == "sum"
    assert isinstance(histories[5], MinScalarRecord)
    assert histories[5].name == "quantile_0.5"
    assert isinstance(histories[6], MinScalarRecord)
    assert histories[6].name == "quantile_0.9"


def test_extended_error_state_reset() -> None:
    state = ExtendedErrorState()
    state.update(torch.arange(6))
    assert state.num_predictions == 6
    state.reset()
    assert state.num_predictions == 0


def test_extended_error_state_update_1d() -> None:
    state = ExtendedErrorState()
    state.update(torch.arange(6))
    assert state.num_predictions == 6


def test_extended_error_state_update_2d() -> None:
    state = ExtendedErrorState()
    state.update(torch.arange(6).view(2, 3))
    assert state.num_predictions == 6


def test_extended_error_state_value_no_quantiles() -> None:
    state = ExtendedErrorState()
    state.update(torch.arange(6))
    assert objects_are_allclose(
        state.value(),
        {
            "mean": 2.5,
            "median": 2,
            "min": 0,
            "max": 5,
            "sum": 15,
            "std": 1.8708287477493286,
            "num_predictions": 6,
        },
    )


def test_extended_error_state_value_with_quantiles() -> None:
    state = ExtendedErrorState(quantiles=[0.5, 0.9])
    state.update(torch.arange(11))
    assert objects_are_allclose(
        state.value(),
        {
            "mean": 5.0,
            "median": 5,
            "min": 0,
            "max": 10,
            "sum": 55,
            "std": 3.316624879837036,
            "quantile_0.5": 5.0,
            "quantile_0.9": 9.0,
            "num_predictions": 11,
        },
    )


def test_extended_error_state_value_track_num_predictions_false() -> None:
    state = ExtendedErrorState(track_num_predictions=False)
    state.update(torch.arange(6))
    assert objects_are_allclose(
        state.value(),
        {
            "mean": 2.5,
            "median": 2,
            "min": 0,
            "max": 5,
            "sum": 15,
            "std": 1.8708287477493286,
        },
    )


@pytest.mark.parametrize("prefix", ["", "prefix_"])
@pytest.mark.parametrize("suffix", ["", "_suffix"])
def test_extended_error_state_value_prefix_suffix(prefix: str, suffix: str) -> None:
    state = ExtendedErrorState(quantiles=[0.5, 0.9])
    state.update(torch.arange(11))
    assert objects_are_allclose(
        state.value(prefix, suffix),
        {
            f"{prefix}mean{suffix}": 5.0,
            f"{prefix}median{suffix}": 5,
            f"{prefix}min{suffix}": 0,
            f"{prefix}max{suffix}": 10,
            f"{prefix}sum{suffix}": 55,
            f"{prefix}std{suffix}": 3.316624879837036,
            f"{prefix}quantile_0.5{suffix}": 5.0,
            f"{prefix}quantile_0.9{suffix}": 9.0,
            f"{prefix}num_predictions{suffix}": 11,
        },
    )


def test_extended_error_state_value_empty() -> None:
    state = ExtendedErrorState()
    with pytest.raises(EmptyMetricError, match="ExtendedErrorState is empty"):
        state.value()
