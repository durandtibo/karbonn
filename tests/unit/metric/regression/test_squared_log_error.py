from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from coola import objects_are_allclose
from coola.utils.tensor import get_available_devices

from karbonn.metric import EmptyMetricError, SquaredAsinhError, SquaredLogError
from karbonn.metric.state import (
    BaseState,
    ErrorState,
    ExtendedErrorState,
    MeanErrorState,
)

MODES = (True, False)
SIZES = (1, 2)
DTYPES = (torch.long, torch.float)


#######################################
#     Tests for SquaredAsinhError     #
#######################################


def test_squared_asinh_error_str() -> None:
    assert str(SquaredAsinhError()).startswith("SquaredAsinhError(")


def test_squared_asinh_error_init_state_default() -> None:
    assert isinstance(SquaredAsinhError().state, ErrorState)


def test_squared_asinh_error_init_state_mean() -> None:
    assert isinstance(SquaredAsinhError(state=MeanErrorState()).state, MeanErrorState)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_squared_asinh_error_forward_correct(
    device: str, mode: bool, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError().to(device=device)
    metric.train(mode)
    metric(
        torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert metric.value() == {
        "mean": 0.0,
        "max": 0.0,
        "min": 0.0,
        "sum": 0.0,
        "num_predictions": batch_size * feature_size,
    }


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_squared_asinh_error_forward_incorrect(
    device: str, mode: bool, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError().to(device=device)
    metric.train(mode)
    metric(
        2 * torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert objects_are_allclose(
        metric.value(),
        {
            "mean": 0.31613843087642435,
            "max": 0.31613843087642435,
            "min": 0.31613843087642435,
            "sum": 0.31613843087642435 * batch_size * feature_size,
            "num_predictions": batch_size * feature_size,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_squared_asinh_error_forward_incorrect_negative(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError().to(device=device)
    metric.train(mode)
    metric(
        -2 * torch.ones(2, 3, device=device),
        -torch.ones(2, 3, device=device),
    )
    assert objects_are_allclose(
        metric.value(),
        {
            "mean": 0.31613843087642435,
            "max": 0.31613843087642435,
            "min": 0.31613843087642435,
            "sum": 1.896830585258546,
            "num_predictions": 6,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_squared_asinh_error_forward_partially_correct(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            "mean": 0.388409699947848,
            "max": 0.776819399895696,
            "min": 0.0,
            "sum": 1.553638799791392,
            "num_predictions": 4,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_squared_asinh_error_forward_1d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, device=device), torch.ones(2, device=device))
    assert metric.value() == {
        "mean": 0.0,
        "max": 0.0,
        "min": 0.0,
        "sum": 0.0,
        "num_predictions": 2,
    }


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_squared_asinh_error_forward_2d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert metric.value() == {
        "mean": 0.0,
        "max": 0.0,
        "min": 0.0,
        "sum": 0.0,
        "num_predictions": 6,
    }


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_squared_asinh_error_forward_3d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 3, 4, device=device), torch.ones(2, 3, 4, device=device))
    assert metric.value() == {
        "mean": 0.0,
        "max": 0.0,
        "min": 0.0,
        "sum": 0.0,
        "num_predictions": 24,
    }


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype_prediction", DTYPES)
@pytest.mark.parametrize("dtype_target", DTYPES)
def test_squared_asinh_error_forward_dtypes(
    device: str,
    mode: bool,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError().to(device=device)
    metric.train(mode)
    metric(
        torch.ones(2, 2, device=device, dtype=dtype_prediction),
        torch.ones(2, 2, device=device, dtype=dtype_target),
    )
    assert metric.value() == {
        "mean": 0.0,
        "max": 0.0,
        "min": 0.0,
        "sum": 0.0,
        "num_predictions": 4,
    }


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_squared_asinh_error_forward_state(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError(state=ExtendedErrorState()).to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    assert metric.value() == {
        "mean": 0.0,
        "max": 0.0,
        "min": 0.0,
        "sum": 0.0,
        "std": 0.0,
        "median": 0.0,
        "num_predictions": 4,
    }


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_squared_asinh_error_forward_multiple_batches(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            "mean": 0.194204849973924,
            "max": 0.776819399895696,
            "min": 0.0,
            "sum": 1.553638799791392,
            "num_predictions": 8,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_squared_asinh_error_forward_multiple_batches_with_reset(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric.reset()
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            "mean": 0.388409699947848,
            "max": 0.776819399895696,
            "min": 0.0,
            "sum": 1.553638799791392,
            "num_predictions": 4,
        },
    )


def test_squared_asinh_error_value_empty() -> None:
    with pytest.raises(EmptyMetricError, match="ErrorState is empty"):
        SquaredAsinhError().value()


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("prefix", ["prefix_", "suffix/"])
@pytest.mark.parametrize("suffix", ["_prefix", "/suffix"])
def test_squared_asinh_error_value_prefix_suffix(device: str, prefix: str, suffix: str) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError().to(device=device)
    metric(torch.ones(2, device=device), torch.ones(2, device=device))
    assert metric.value(prefix, suffix) == {
        f"{prefix}mean{suffix}": 0.0,
        f"{prefix}max{suffix}": 0.0,
        f"{prefix}min{suffix}": 0.0,
        f"{prefix}sum{suffix}": 0.0,
        f"{prefix}num_predictions{suffix}": 2,
    }


def test_squared_asinh_error_reset() -> None:
    state = Mock(spec=BaseState)
    metric = SquaredAsinhError(state=state)
    metric.reset()
    state.reset.assert_called_once_with()


#####################################
#     Tests for SquaredLogError     #
#####################################


def test_squared_log_error_str() -> None:
    assert str(SquaredLogError()).startswith("SquaredLogError(")


def test_squared_log_error_init_state_default() -> None:
    assert isinstance(SquaredLogError().state, ErrorState)


def test_squared_log_error_init_state_mean() -> None:
    assert isinstance(SquaredLogError(state=MeanErrorState()).state, MeanErrorState)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_squared_log_error_forward_correct(
    device: str, mode: bool, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = SquaredLogError().to(device=device)
    metric.train(mode)
    metric(
        torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert metric.value() == {
        "mean": 0.0,
        "max": 0.0,
        "min": 0.0,
        "sum": 0.0,
        "num_predictions": batch_size * feature_size,
    }


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_squared_log_error_forward_incorrect(
    device: str, mode: bool, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = SquaredLogError().to(device=device)
    metric.train(mode)
    metric(
        2 * torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert objects_are_allclose(
        metric.value(),
        {
            "mean": 0.16440195389316548,
            "max": 0.16440195389316548,
            "min": 0.16440195389316548,
            "sum": 0.16440195389316548 * batch_size * feature_size,
            "num_predictions": batch_size * feature_size,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_squared_log_error_forward_partially_correct(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = SquaredLogError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            "mean": 0.2402265069591007,
            "max": 0.4804530139182014,
            "min": 0.0,
            "sum": 0.9609060278364028,
            "num_predictions": 4,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_squared_log_error_forward_1d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = SquaredLogError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, device=device), torch.ones(2, device=device))
    assert metric.value() == {
        "mean": 0.0,
        "max": 0.0,
        "min": 0.0,
        "sum": 0.0,
        "num_predictions": 2,
    }


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_squared_log_error_forward_2d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = SquaredLogError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert metric.value() == {
        "mean": 0.0,
        "max": 0.0,
        "min": 0.0,
        "sum": 0.0,
        "num_predictions": 6,
    }


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_squared_log_error_forward_3d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = SquaredLogError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 3, 4, device=device), torch.ones(2, 3, 4, device=device))
    assert metric.value() == {
        "mean": 0.0,
        "max": 0.0,
        "min": 0.0,
        "sum": 0.0,
        "num_predictions": 24,
    }


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype_prediction", DTYPES)
@pytest.mark.parametrize("dtype_target", DTYPES)
def test_squared_log_error_forward_dtypes(
    device: str,
    mode: bool,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    if device == "mps:0" and (dtype_prediction != torch.float or dtype_target != torch.float):
        return  # MPS does not support log1p op with int64 input
    device = torch.device(device)
    metric = SquaredLogError().to(device=device)
    metric.train(mode)
    metric(
        torch.ones(2, 2, device=device, dtype=dtype_prediction),
        torch.ones(2, 2, device=device, dtype=dtype_target),
    )
    assert metric.value() == {
        "mean": 0.0,
        "max": 0.0,
        "min": 0.0,
        "sum": 0.0,
        "num_predictions": 4,
    }


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_squared_log_error_forward_state(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = SquaredLogError(state=ExtendedErrorState()).to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    assert metric.value() == {
        "mean": 0.0,
        "max": 0.0,
        "min": 0.0,
        "sum": 0.0,
        "std": 0.0,
        "median": 0.0,
        "num_predictions": 4,
    }


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_squared_log_error_forward_multiple_batches(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = SquaredLogError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            "mean": 0.12011325347955035,
            "max": 0.4804530139182014,
            "min": 0.0,
            "sum": 0.9609060278364028,
            "num_predictions": 8,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_squared_log_error_forward_multiple_batches_with_reset(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = SquaredLogError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric.reset()
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            "mean": 0.2402265069591007,
            "max": 0.4804530139182014,
            "min": 0.0,
            "sum": 0.9609060278364028,
            "num_predictions": 4,
        },
    )


def test_squared_log_error_value_empty() -> None:
    with pytest.raises(EmptyMetricError, match="ErrorState is empty"):
        SquaredLogError().value()


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("prefix", ["prefix_", "suffix/"])
@pytest.mark.parametrize("suffix", ["_prefix", "/suffix"])
def test_squared_log_error_value_prefix_suffix(device: str, prefix: str, suffix: str) -> None:
    device = torch.device(device)
    metric = SquaredLogError().to(device=device)
    metric(torch.ones(2, device=device), torch.ones(2, device=device))
    assert metric.value(prefix, suffix) == {
        f"{prefix}mean{suffix}": 0.0,
        f"{prefix}max{suffix}": 0.0,
        f"{prefix}min{suffix}": 0.0,
        f"{prefix}sum{suffix}": 0.0,
        f"{prefix}num_predictions{suffix}": 2,
    }


def test_squared_log_error_reset() -> None:
    state = Mock(spec=BaseState)
    metric = SquaredLogError(state=state)
    metric.reset()
    state.reset.assert_called_once_with()
