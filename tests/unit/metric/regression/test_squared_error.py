from __future__ import annotations

import math
from unittest.mock import Mock

import pytest
import torch
from coola import objects_are_allclose
from coola.utils.tensor import get_available_devices

from karbonn.metric import EmptyMetricError, RootMeanSquaredError, SquaredError
from karbonn.metric.state import (
    BaseState,
    ErrorState,
    ExtendedErrorState,
    MeanErrorState,
)

MODES = (True, False)
SIZES = (1, 2)
DTYPES = (torch.long, torch.float)


##################################
#     Tests for SquaredError     #
##################################


def test_squared_error_str() -> None:
    assert str(SquaredError()).startswith("SquaredError(")


def test_squared_error_init_state_default() -> None:
    assert isinstance(SquaredError().state, ErrorState)


def test_squared_error_init_state_mean() -> None:
    assert isinstance(SquaredError(state=MeanErrorState()).state, MeanErrorState)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_squared_error_forward_correct(
    device: str, mode: bool, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = SquaredError().to(device=device)
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
def test_squared_error_forward_incorrect(
    device: str, mode: bool, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = SquaredError().to(device=device)
    metric.train(mode)
    metric(
        -torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert metric.value() == {
        "mean": 4.0,
        "max": 4.0,
        "min": 4.0,
        "sum": 4.0 * batch_size * feature_size,
        "num_predictions": batch_size * feature_size,
    }


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_squared_error_forward_partially_correct(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = SquaredError().to(device=device)
    metric.train(mode)
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    assert metric.value() == {
        "mean": 2.0,
        "max": 4.0,
        "min": 0.0,
        "sum": 8.0,
        "num_predictions": 4,
    }


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_squared_error_forward_1d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = SquaredError().to(device=device)
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
def test_squared_error_forward_2d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = SquaredError().to(device=device)
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
def test_squared_error_forward_3d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = SquaredError().to(device=device)
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
def test_squared_error_forward_dtypes(
    device: str,
    mode: bool,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = SquaredError().to(device=device)
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
def test_squared_error_forward_state(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = SquaredError(state=ExtendedErrorState()).to(device=device)
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
def test_squared_error_forward_multiple_batches(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = SquaredError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    assert metric.value() == {
        "mean": 1.0,
        "max": 4.0,
        "min": 0.0,
        "sum": 8.0,
        "num_predictions": 8,
    }


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_squared_error_forward_multiple_batches_with_reset(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = SquaredError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric.reset()
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    assert metric.value() == {
        "mean": 2.0,
        "max": 4.0,
        "min": 0.0,
        "sum": 8.0,
        "num_predictions": 4,
    }


def test_squared_error_value_empty() -> None:
    with pytest.raises(EmptyMetricError, match="ErrorState is empty"):
        SquaredError().value()


def test_squared_error_reset() -> None:
    state = Mock(spec=BaseState)
    metric = SquaredError(state=state)
    metric.reset()
    state.reset.assert_called_once_with()


##########################################
#     Tests for RootMeanSquaredError     #
##########################################


def test_root_mean_squared_error_str() -> None:
    assert str(RootMeanSquaredError()).startswith("RootMeanSquaredError(")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_root_mean_squared_error_forward_correct(
    device: str, mode: bool, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = RootMeanSquaredError().to(device=device)
    metric.train(mode)
    metric(
        torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert metric.value() == {
        "mean": 0.0,
        "num_predictions": batch_size * feature_size,
    }


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_root_mean_squared_error_forward_incorrect(
    device: str, mode: bool, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = RootMeanSquaredError().to(device=device)
    metric.train(mode)
    metric(
        torch.ones(batch_size, feature_size, device=device),
        -torch.ones(batch_size, feature_size, device=device),
    )
    assert metric.value() == {
        "mean": 2.0,
        "num_predictions": batch_size * feature_size,
    }


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_root_mean_squared_error_forward_partially_correct(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = RootMeanSquaredError().to(device=device)
    metric.train(mode)
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            "mean": math.sqrt(2.0),
            "num_predictions": 4,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_root_mean_squared_error_forward_1d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = RootMeanSquaredError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, device=device), torch.ones(2, device=device))
    assert metric.value() == {
        "mean": 0.0,
        "num_predictions": 2,
    }


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_root_mean_squared_error_forward_2d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = RootMeanSquaredError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert metric.value() == {
        "mean": 0.0,
        "num_predictions": 6,
    }


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_root_mean_squared_error_forward_3d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = RootMeanSquaredError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 3, 4, device=device), torch.ones(2, 3, 4, device=device))
    assert metric.value() == {
        "mean": 0.0,
        "num_predictions": 24,
    }


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype_prediction", DTYPES)
@pytest.mark.parametrize("dtype_target", DTYPES)
def test_root_mean_squared_error_forward_dtypes(
    device: str,
    mode: bool,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = RootMeanSquaredError().to(device=device)
    metric.train(mode)
    metric(
        torch.ones(2, 2, device=device, dtype=dtype_prediction),
        torch.ones(2, 2, device=device, dtype=dtype_target),
    )
    assert metric.value() == {
        "mean": 0.0,
        "num_predictions": 4,
    }


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_root_mean_squared_error_forward_multiple_batches(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = RootMeanSquaredError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    assert metric.value() == {
        "mean": 1.0,
        "num_predictions": 8,
    }


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_root_mean_squared_error_forward_multiple_batches_with_reset(
    device: str, mode: bool
) -> None:
    device = torch.device(device)
    metric = RootMeanSquaredError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric.reset()
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            "mean": math.sqrt(2.0),
            "num_predictions": 4,
        },
    )


def test_root_mean_squared_error_value_empty() -> None:
    with pytest.raises(EmptyMetricError, match="ErrorState is empty"):
        RootMeanSquaredError().value()


def test_root_mean_squared_error_reset() -> None:
    state = Mock(spec=BaseState)
    metric = RootMeanSquaredError(state=state)
    metric.reset()
    state.reset.assert_called_once_with()
