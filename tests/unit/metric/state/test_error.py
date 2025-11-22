from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal
from minrecord import MinScalarRecord

from karbonn.metric import EmptyMetricError
from karbonn.metric.state import (
    ErrorState,
    ExtendedErrorState,
    MeanErrorState,
    NormalizedMeanSquaredErrorState,
    RootMeanErrorState,
)
from karbonn.utils.tracker import (
    MeanTensorTracker,
    ScalableTensorTracker,
    TensorTracker,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

################################
#     Tests for ErrorState     #
################################


def test_error_state_repr() -> None:
    assert repr(ErrorState()).startswith("ErrorState(")


def test_error_state_str() -> None:
    assert str(ErrorState()).startswith("ErrorState(")


def test_error_state_clone() -> None:
    state = ErrorState(ScalableTensorTracker(count=4, total=10.0, min_value=0.0, max_value=5.0))
    cloned = state.clone()
    assert state is not cloned
    assert state.equal(cloned)
    state.update(torch.ones(4))
    assert not state.equal(cloned)


def test_error_state_clone_empty() -> None:
    state = ErrorState()
    cloned = state.clone()
    assert state is not cloned
    assert state.equal(cloned)
    state.update(torch.ones(4))
    assert not state.equal(cloned)


def test_error_state_equal_true() -> None:
    assert ErrorState(
        ScalableTensorTracker(count=4, total=10.0, min_value=0.0, max_value=5.0)
    ).equal(ErrorState(ScalableTensorTracker(count=4, total=10.0, min_value=0.0, max_value=5.0)))


def test_error_state_equal_true_empty() -> None:
    assert ErrorState().equal(ErrorState())


def test_error_state_equal_false_different_tracker() -> None:
    assert not ErrorState(
        ScalableTensorTracker(count=4, total=10.0, min_value=0.0, max_value=5.0)
    ).equal(ErrorState())


def test_error_state_equal_false_different_track_count() -> None:
    assert not ErrorState().equal(ErrorState(track_count=False))


def test_error_state_equal_false_different_type() -> None:
    assert not ErrorState().equal(ScalableTensorTracker())


def test_error_state_get_records() -> None:
    assert objects_are_equal(
        ErrorState().get_records(),
        (
            MinScalarRecord(name="mean"),
            MinScalarRecord(name="min"),
            MinScalarRecord(name="max"),
            MinScalarRecord(name="sum"),
        ),
    )


@pytest.mark.parametrize("prefix", ["", "prefix_"])
@pytest.mark.parametrize("suffix", ["", "_suffix"])
def test_error_state_get_records_prefix_suffix(prefix: str, suffix: str) -> None:
    assert objects_are_equal(
        ErrorState().get_records(prefix, suffix),
        (
            MinScalarRecord(name=f"{prefix}mean{suffix}"),
            MinScalarRecord(name=f"{prefix}min{suffix}"),
            MinScalarRecord(name=f"{prefix}max{suffix}"),
            MinScalarRecord(name=f"{prefix}sum{suffix}"),
        ),
    )


def test_error_state_reset() -> None:
    state = ErrorState()
    state.update(torch.arange(6))
    assert state.count == 6
    state.reset()
    assert state.count == 0


def test_error_state_update_1d() -> None:
    state = ErrorState()
    state.update(torch.arange(6))
    assert state._tracker.equal(
        ScalableTensorTracker(count=6, total=15.0, min_value=0.0, max_value=5.0)
    )


def test_error_state_update_2d() -> None:
    state = ErrorState()
    state.update(torch.arange(6).view(2, 3))
    assert state._tracker.equal(
        ScalableTensorTracker(count=6, total=15.0, min_value=0.0, max_value=5.0)
    )


def test_error_state_value() -> None:
    state = ErrorState()
    state.update(torch.arange(6))
    assert objects_are_equal(
        state.value(),
        {
            "mean": 2.5,
            "min": 0.0,
            "max": 5.0,
            "sum": 15.0,
            "count": 6,
        },
    )


def test_error_state_value_count_false() -> None:
    state = ErrorState(track_count=False)
    state.update(torch.arange(6))
    assert objects_are_equal(state.value(), {"mean": 2.5, "min": 0.0, "max": 5.0, "sum": 15.0})


@pytest.mark.parametrize("prefix", ["", "prefix_"])
@pytest.mark.parametrize("suffix", ["", "_suffix"])
def test_error_state_value_prefix_suffix(prefix: str, suffix: str) -> None:
    state = ErrorState()
    state.update(torch.arange(6))
    assert objects_are_equal(
        state.value(prefix, suffix),
        {
            f"{prefix}mean{suffix}": 2.5,
            f"{prefix}min{suffix}": 0.0,
            f"{prefix}max{suffix}": 5.0,
            f"{prefix}sum{suffix}": 15.0,
            f"{prefix}count{suffix}": 6,
        },
    )


def test_error_state_value_empty() -> None:
    state = ErrorState()
    with pytest.raises(EmptyMetricError, match=r"ErrorState is empty"):
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


def test_extended_error_state_clone() -> None:
    state = ExtendedErrorState(tracker=TensorTracker(torch.arange(6)))
    cloned = state.clone()
    assert state is not cloned
    assert state.equal(cloned)
    state.update(torch.ones(4))
    assert not state.equal(cloned)


def test_extended_error_state_clone_empty() -> None:
    state = ExtendedErrorState()
    cloned = state.clone()
    assert state is not cloned
    assert state.equal(cloned)
    state.update(torch.ones(4))
    assert not state.equal(cloned)


def test_extended_error_state_equal_true() -> None:
    assert ExtendedErrorState(tracker=TensorTracker(torch.arange(6))).equal(
        ExtendedErrorState(tracker=TensorTracker(torch.arange(6)))
    )


def test_extended_error_state_equal_true_empty() -> None:
    assert ExtendedErrorState().equal(ExtendedErrorState())


def test_extended_error_state_equal_false_different_tracker() -> None:
    assert not ExtendedErrorState(tracker=TensorTracker(torch.arange(6))).equal(
        ExtendedErrorState()
    )


def test_extended_error_state_equal_false_different_track_count() -> None:
    assert not ExtendedErrorState().equal(ExtendedErrorState(track_count=False))


def test_extended_error_state_equal_false_different_type() -> None:
    assert not ExtendedErrorState().equal(ScalableTensorTracker())


def test_extended_error_state_get_records_no_quantile() -> None:
    assert objects_are_equal(
        ExtendedErrorState().get_records(),
        (
            MinScalarRecord(name="mean"),
            MinScalarRecord(name="median"),
            MinScalarRecord(name="min"),
            MinScalarRecord(name="max"),
            MinScalarRecord(name="sum"),
        ),
    )


def test_extended_error_state_get_records_quantiles() -> None:
    assert objects_are_equal(
        ExtendedErrorState(quantiles=[0.5, 0.9]).get_records(),
        (
            MinScalarRecord(name="mean"),
            MinScalarRecord(name="median"),
            MinScalarRecord(name="min"),
            MinScalarRecord(name="max"),
            MinScalarRecord(name="sum"),
            MinScalarRecord(name="quantile_0.5"),
            MinScalarRecord(name="quantile_0.9"),
        ),
    )


def test_extended_error_state_reset() -> None:
    state = ExtendedErrorState()
    state.update(torch.arange(6))
    assert state.count == 6
    state.reset()
    assert state.count == 0


def test_extended_error_state_update_1d() -> None:
    state = ExtendedErrorState()
    state.update(torch.arange(6))
    assert state.count == 6


def test_extended_error_state_update_2d() -> None:
    state = ExtendedErrorState()
    state.update(torch.arange(6).view(2, 3))
    assert state.count == 6


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
            "count": 6,
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
            "count": 11,
        },
    )


def test_extended_error_state_value_track_count_false() -> None:
    state = ExtendedErrorState(track_count=False)
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
            f"{prefix}count{suffix}": 11,
        },
    )


def test_extended_error_state_value_empty() -> None:
    state = ExtendedErrorState()
    with pytest.raises(EmptyMetricError, match=r"ExtendedErrorState is empty"):
        state.value()


####################################
#     Tests for MeanErrorState     #
####################################


def test_mean_error_state_repr() -> None:
    assert repr(MeanErrorState()).startswith("MeanErrorState(")


def test_mean_error_state_str() -> None:
    assert str(MeanErrorState()).startswith("MeanErrorState(")


def test_mean_error_state_clone() -> None:
    state = MeanErrorState(MeanTensorTracker(count=4, total=10.0))
    cloned = state.clone()
    assert state is not cloned
    assert state.equal(cloned)
    state.update(torch.ones(4))
    assert not state.equal(cloned)


def test_mean_error_state_clone_empty() -> None:
    state = MeanErrorState()
    cloned = state.clone()
    assert state is not cloned
    assert state.equal(cloned)
    state.update(torch.ones(4))
    assert not state.equal(cloned)


def test_mean_error_state_equal_true() -> None:
    assert MeanErrorState(MeanTensorTracker(count=4, total=10.0)).equal(
        MeanErrorState(MeanTensorTracker(count=4, total=10.0))
    )


def test_mean_error_state_equal_true_empty() -> None:
    assert MeanErrorState().equal(MeanErrorState())


def test_mean_error_state_equal_false_different_tracker() -> None:
    assert not MeanErrorState(MeanTensorTracker(count=4, total=10.0)).equal(MeanErrorState())


def test_mean_error_state_equal_false_different_track_count() -> None:
    assert not MeanErrorState().equal(MeanErrorState(track_count=False))


def test_mean_error_state_equal_false_different_type() -> None:
    assert not MeanErrorState().equal(ScalableTensorTracker())


def test_mean_error_state_get_records() -> None:
    assert objects_are_equal(
        MeanErrorState().get_records(),
        (MinScalarRecord(name="mean"),),
    )


@pytest.mark.parametrize("prefix", ["", "prefix_"])
@pytest.mark.parametrize("suffix", ["", "_suffix"])
def test_mean_error_state_get_records_prefix_suffix(prefix: str, suffix: str) -> None:
    assert objects_are_equal(
        MeanErrorState().get_records(prefix, suffix),
        (MinScalarRecord(name=f"{prefix}mean{suffix}"),),
    )


def test_mean_error_state_reset() -> None:
    state = MeanErrorState()
    state.update(torch.arange(6))
    assert state.count == 6
    state.reset()
    assert state.count == 0


def test_mean_error_state_update_1d() -> None:
    state = MeanErrorState()
    state.update(torch.arange(6))
    assert state._tracker.equal(MeanTensorTracker(count=6, total=15.0))


def test_mean_error_state_update_2d() -> None:
    state = MeanErrorState()
    state.update(torch.arange(6).view(2, 3))
    assert state._tracker.equal(MeanTensorTracker(count=6, total=15.0))


def test_mean_error_state_value() -> None:
    state = MeanErrorState()
    state.update(torch.arange(6))
    assert objects_are_equal(state.value(), {"mean": 2.5, "count": 6})


def test_mean_error_state_value_correct() -> None:
    state = MeanErrorState()
    state.update(torch.zeros(4))
    assert objects_are_equal(state.value(), {"mean": 0.0, "count": 4})


def test_mean_error_state_value_track_count_false() -> None:
    state = MeanErrorState(track_count=False)
    state.update(torch.arange(6))
    assert objects_are_equal(state.value(), {"mean": 2.5})


@pytest.mark.parametrize("prefix", ["", "prefix_"])
@pytest.mark.parametrize("suffix", ["", "_suffix"])
def test_mean_error_state_value_prefix_suffix(prefix: str, suffix: str) -> None:
    state = MeanErrorState()
    state.update(torch.arange(6))
    assert objects_are_equal(
        state.value(prefix, suffix),
        {
            f"{prefix}mean{suffix}": 2.5,
            f"{prefix}count{suffix}": 6,
        },
    )


def test_mean_error_state_value_empty() -> None:
    state = MeanErrorState()
    with pytest.raises(EmptyMetricError, match=r"MeanErrorState is empty"):
        state.value()


########################################
#     Tests for RootMeanErrorState     #
########################################


def test_root_mean_error_state_repr() -> None:
    assert repr(RootMeanErrorState()).startswith("RootMeanErrorState(")


def test_root_mean_error_state_str() -> None:
    assert str(RootMeanErrorState()).startswith("RootMeanErrorState(")


def test_root_mean_error_state_clone() -> None:
    state = RootMeanErrorState(MeanTensorTracker(count=4, total=10.0))
    cloned = state.clone()
    assert state is not cloned
    assert state.equal(cloned)
    state.update(torch.ones(4))
    assert not state.equal(cloned)


def test_root_mean_error_state_clone_empty() -> None:
    state = RootMeanErrorState()
    cloned = state.clone()
    assert state is not cloned
    assert state.equal(cloned)
    state.update(torch.ones(4))
    assert not state.equal(cloned)


def test_root_mean_error_state_equal_true() -> None:
    assert RootMeanErrorState(MeanTensorTracker(count=4, total=10.0)).equal(
        RootMeanErrorState(MeanTensorTracker(count=4, total=10.0))
    )


def test_root_mean_error_state_equal_true_empty() -> None:
    assert RootMeanErrorState().equal(RootMeanErrorState())


def test_root_mean_error_state_equal_false_different_tracker() -> None:
    assert not RootMeanErrorState(MeanTensorTracker(count=4, total=10.0)).equal(
        RootMeanErrorState()
    )


def test_root_mean_error_state_equal_false_different_track_count() -> None:
    assert not RootMeanErrorState().equal(RootMeanErrorState(track_count=False))


def test_root_mean_error_state_equal_false_different_type() -> None:
    assert not RootMeanErrorState().equal(ScalableTensorTracker())


def test_root_mean_error_state_get_records() -> None:
    assert objects_are_equal(
        RootMeanErrorState().get_records(),
        (MinScalarRecord(name="mean"),),
    )


@pytest.mark.parametrize("prefix", ["", "prefix_"])
@pytest.mark.parametrize("suffix", ["", "_suffix"])
def test_root_mean_error_state_get_records_prefix_suffix(prefix: str, suffix: str) -> None:
    assert objects_are_equal(
        RootMeanErrorState().get_records(prefix, suffix),
        (MinScalarRecord(name=f"{prefix}mean{suffix}"),),
    )


def test_root_mean_error_state_reset() -> None:
    state = RootMeanErrorState()
    state.update(torch.arange(6))
    assert state.count == 6
    state.reset()
    assert state.count == 0


def test_root_mean_error_state_update_1d() -> None:
    state = RootMeanErrorState()
    state.update(torch.arange(6))
    assert state._tracker.equal(MeanTensorTracker(count=6, total=15.0))


def test_root_mean_error_state_update_2d() -> None:
    state = RootMeanErrorState()
    state.update(torch.arange(6).view(2, 3))
    assert state._tracker.equal(MeanTensorTracker(count=6, total=15.0))


def test_root_mean_error_state_value() -> None:
    state = RootMeanErrorState()
    state.update(torch.tensor([1, 9, 2, 7, 3, 2]))
    assert objects_are_equal(state.value(), {"mean": 2.0, "count": 6})


def test_root_mean_error_state_value_correct() -> None:
    state = RootMeanErrorState()
    state.update(torch.zeros(4))
    assert objects_are_equal(state.value(), {"mean": 0.0, "count": 4})


def test_root_mean_error_state_value_track_count_false() -> None:
    state = RootMeanErrorState(track_count=False)
    state.update(torch.tensor([1, 9, 2, 7, 3, 2]))
    assert objects_are_equal(state.value(), {"mean": 2.0})


@pytest.mark.parametrize("prefix", ["", "prefix_"])
@pytest.mark.parametrize("suffix", ["", "_suffix"])
def test_root_mean_error_state_value_prefix_suffix(prefix: str, suffix: str) -> None:
    state = RootMeanErrorState()
    state.update(torch.tensor([1, 9, 2, 7, 3, 2]))
    assert objects_are_equal(
        state.value(prefix, suffix),
        {
            f"{prefix}mean{suffix}": 2.0,
            f"{prefix}count{suffix}": 6,
        },
    )


def test_root_mean_error_state_value_empty() -> None:
    state = RootMeanErrorState()
    with pytest.raises(EmptyMetricError, match=r"RootMeanErrorState is empty"):
        state.value()


#####################################################
#     Tests for NormalizedMeanSquaredErrorState     #
#####################################################


def test_normalized_mean_squared_error_state_repr() -> None:
    assert repr(NormalizedMeanSquaredErrorState()).startswith("NormalizedMeanSquaredErrorState(")


def test_normalized_mean_squared_error_state_str() -> None:
    assert str(NormalizedMeanSquaredErrorState()).startswith("NormalizedMeanSquaredErrorState(")


def test_normalized_mean_squared_error_state_clone() -> None:
    state = NormalizedMeanSquaredErrorState(
        squared_errors=MeanTensorTracker(count=4, total=10.0),
        squared_targets=MeanTensorTracker(count=4, total=16.0),
    )
    cloned = state.clone()
    assert state is not cloned
    assert state.equal(cloned)
    state.update(torch.ones(4), torch.ones(4))
    assert not state.equal(cloned)


def test_normalized_mean_squared_error_state_clone_empty() -> None:
    state = NormalizedMeanSquaredErrorState()
    cloned = state.clone()
    assert state is not cloned
    assert state.equal(cloned)
    state.update(torch.ones(4), torch.ones(4))
    assert not state.equal(cloned)


def test_normalized_mean_squared_error_state_equal_true() -> None:
    assert NormalizedMeanSquaredErrorState(
        squared_errors=MeanTensorTracker(count=4, total=10.0),
        squared_targets=MeanTensorTracker(count=4, total=16.0),
    ).equal(
        NormalizedMeanSquaredErrorState(
            squared_errors=MeanTensorTracker(count=4, total=10.0),
            squared_targets=MeanTensorTracker(count=4, total=16.0),
        )
    )


def test_normalized_mean_squared_error_state_equal_true_empty() -> None:
    assert NormalizedMeanSquaredErrorState().equal(NormalizedMeanSquaredErrorState())


def test_normalized_mean_squared_error_state_equal_false_different_errors() -> None:
    assert not NormalizedMeanSquaredErrorState(
        squared_errors=MeanTensorTracker(count=4, total=10.0)
    ).equal(NormalizedMeanSquaredErrorState())


def test_normalized_mean_squared_error_state_equal_false_different_targets() -> None:
    assert not NormalizedMeanSquaredErrorState(
        squared_targets=MeanTensorTracker(count=4, total=10.0)
    ).equal(NormalizedMeanSquaredErrorState())


def test_normalized_mean_squared_error_state_equal_false_different_track_count() -> None:
    assert not NormalizedMeanSquaredErrorState().equal(
        NormalizedMeanSquaredErrorState(track_count=False)
    )


def test_normalized_mean_squared_error_state_equal_false_different_type() -> None:
    assert not NormalizedMeanSquaredErrorState().equal(ScalableTensorTracker())


def test_normalized_mean_squared_error_state_get_records() -> None:
    assert objects_are_equal(
        NormalizedMeanSquaredErrorState().get_records(),
        (MinScalarRecord(name="mean"),),
    )


@pytest.mark.parametrize("prefix", ["", "prefix_"])
@pytest.mark.parametrize("suffix", ["", "_suffix"])
def test_normalized_mean_squared_error_state_get_records_prefix_suffix(
    prefix: str, suffix: str
) -> None:
    assert objects_are_equal(
        NormalizedMeanSquaredErrorState().get_records(prefix, suffix),
        (MinScalarRecord(name=f"{prefix}mean{suffix}"),),
    )


def test_normalized_mean_squared_error_state_reset() -> None:
    state = NormalizedMeanSquaredErrorState()
    state.update(torch.arange(6), torch.arange(6))
    assert state.count == 6
    state.reset()
    assert state.count == 0


def test_normalized_mean_squared_error_state_update_1d() -> None:
    state = NormalizedMeanSquaredErrorState()
    state.update(torch.arange(6), torch.ones(6))
    assert state._squared_errors.equal(MeanTensorTracker(total=55.0, count=6))
    assert state._squared_targets.equal(MeanTensorTracker(total=6.0, count=6))


def test_normalized_mean_squared_error_state_update_2d() -> None:
    state = NormalizedMeanSquaredErrorState()
    state.update(torch.arange(6).view(2, 3), torch.ones(2, 3))
    assert state._squared_errors.equal(MeanTensorTracker(total=55.0, count=6))
    assert state._squared_targets.equal(MeanTensorTracker(total=6.0, count=6))


def test_normalized_mean_squared_error_state_value() -> None:
    state = NormalizedMeanSquaredErrorState()
    state.update(torch.tensor([1, 9, 2, 7, 3, 2]), torch.tensor([1, 2, 3, 4, 5, 6]))
    assert objects_are_allclose(state.value(), {"mean": 1.6263736486434937, "count": 6})


def test_normalized_mean_squared_error_state_value_correct() -> None:
    state = NormalizedMeanSquaredErrorState()
    state.update(torch.zeros(4), torch.ones(4))
    assert objects_are_equal(state.value(), {"mean": 0.0, "count": 4})


def test_normalized_mean_squared_error_state_value_track_count_false() -> None:
    state = NormalizedMeanSquaredErrorState(track_count=False)
    state.update(torch.zeros(4), torch.ones(4))
    assert objects_are_equal(state.value(), {"mean": 0.0})


@pytest.mark.parametrize("prefix", ["", "prefix_"])
@pytest.mark.parametrize("suffix", ["", "_suffix"])
def test_normalized_mean_squared_error_state_value_prefix_suffix(prefix: str, suffix: str) -> None:
    state = NormalizedMeanSquaredErrorState()
    state.update(torch.zeros(4), torch.ones(4))
    assert objects_are_equal(
        state.value(prefix, suffix),
        {
            f"{prefix}mean{suffix}": 0.0,
            f"{prefix}count{suffix}": 4,
        },
    )


def test_normalized_mean_squared_error_state_value_empty() -> None:
    state = NormalizedMeanSquaredErrorState()
    with pytest.raises(EmptyMetricError, match=r"NormalizedMeanSquaredErrorState is empty"):
        state.value()
