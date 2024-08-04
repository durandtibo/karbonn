from unittest.mock import Mock

import pytest
import torch
from coola import objects_are_equal
from coola.utils.tensor import get_available_devices

from karbonn.metric import AbsoluteRelativeError, EmptyMetricError
from karbonn.metric.state import (
    BaseState,
    ErrorState,
    ExtendedErrorState,
    MeanErrorState,
)

MODES = (True, False)
SIZES = (1, 2)
DTYPES = (torch.long, torch.float)


###########################################
#     Tests for AbsoluteRelativeError     #
###########################################


def test_absolute_relative_error_str() -> None:
    assert str(AbsoluteRelativeError()).startswith("AbsoluteRelativeError(")


@pytest.mark.parametrize("eps", [1e-5, 0.1])
def test_absolute_relative_error_init_eps(eps: float) -> None:
    assert AbsoluteRelativeError(eps=eps)._eps == eps


def test_absolute_relative_error_init_eps_default() -> None:
    assert AbsoluteRelativeError()._eps == 1e-8


@pytest.mark.parametrize("eps", [0.0, -0.1])
def test_absolute_relative_error_init_eps_incorrect(eps: float) -> None:
    with pytest.raises(ValueError, match="Incorrect eps"):
        AbsoluteRelativeError(eps=eps)


def test_absolute_relative_error_init_state_default() -> None:
    assert isinstance(AbsoluteRelativeError()._state, ErrorState)


def test_absolute_relative_error_init_state_mean() -> None:
    assert isinstance(AbsoluteRelativeError(state=MeanErrorState())._state, MeanErrorState)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_absolute_relative_error_forward_correct(
    device: str, mode: bool, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = AbsoluteRelativeError().to(device=device)
    metric.train(mode)
    metric(
        torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert objects_are_equal(
        metric.value(),
        {
            "mean": 0.0,
            "max": 0.0,
            "min": 0.0,
            "sum": 0.0,
            "num_predictions": batch_size * feature_size,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_absolute_relative_error_forward_correct_zero(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = AbsoluteRelativeError().to(device=device)
    metric.train(mode)
    metric(torch.zeros(2, 2, device=device), torch.zeros(2, 2, device=device))
    assert objects_are_equal(
        metric.value(),
        {
            "mean": 0.0,
            "max": 0.0,
            "min": 0.0,
            "sum": 0.0,
            "num_predictions": 4,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_absolute_relative_error_forward_incorrect(
    device: str, mode: bool, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = AbsoluteRelativeError().to(device=device)
    metric.train(mode)
    metric(
        torch.ones(batch_size, feature_size, device=device).mul(2),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert objects_are_equal(
        metric.value(),
        {
            "mean": 1.0,
            "max": 1.0,
            "min": 1.0,
            "sum": float(batch_size * feature_size),
            "num_predictions": batch_size * feature_size,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_absolute_relative_error_forward_partially_correct_zero_prediction(
    device: str, mode: bool
) -> None:
    device = torch.device(device)
    metric = AbsoluteRelativeError().to(device=device)
    metric.train(mode)
    metric(torch.eye(2, device=device), torch.ones(2, 2, device=device))
    assert objects_are_equal(
        metric.value(),
        {
            "mean": 0.5,
            "max": 1.0,
            "min": 0.0,
            "sum": 2.0,
            "num_predictions": 4,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_absolute_relative_error_forward_partially_correct_zero_target(
    device: str, mode: bool
) -> None:
    device = torch.device(device)
    metric = AbsoluteRelativeError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device))
    assert objects_are_equal(
        metric.value(),
        {
            "mean": 5e7,
            "max": 1e8,
            "min": 0.0,
            "sum": 2e8,
            "num_predictions": 4,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_absolute_relative_error_forward_1d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = AbsoluteRelativeError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, device=device), torch.ones(2, device=device))
    assert objects_are_equal(
        metric.value(),
        {
            "mean": 0.0,
            "max": 0.0,
            "min": 0.0,
            "sum": 0.0,
            "num_predictions": 2,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_absolute_relative_error_forward_3d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = AbsoluteRelativeError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 3, 4, device=device), torch.ones(2, 3, 4, device=device))
    assert objects_are_equal(
        metric.value(),
        {
            "mean": 0.0,
            "max": 0.0,
            "min": 0.0,
            "sum": 0.0,
            "num_predictions": 24,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype_prediction", DTYPES)
@pytest.mark.parametrize("dtype_target", DTYPES)
def test_absolute_relative_error_forward_dtype(
    device: str,
    mode: bool,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = AbsoluteRelativeError(eps=1).to(device=device)
    metric.train(mode)
    metric(
        torch.ones(2, 3, device=device, dtype=dtype_prediction),
        torch.ones(2, 3, device=device, dtype=dtype_target),
    )
    assert objects_are_equal(
        metric.value(),
        {
            "mean": 0.0,
            "max": 0.0,
            "min": 0.0,
            "sum": 0.0,
            "num_predictions": 6,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_absolute_relative_error_forward_state(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = AbsoluteRelativeError(state=ExtendedErrorState()).to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    assert objects_are_equal(
        metric.value(),
        {
            "mean": 0.0,
            "max": 0.0,
            "min": 0.0,
            "sum": 0.0,
            "std": 0.0,
            "median": 0.0,
            "num_predictions": 4,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_absolute_relative_error_forward_multiple_batches(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = AbsoluteRelativeError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device) + 1)
    assert objects_are_equal(
        metric.value(),
        {
            "mean": 0.125,
            "max": 0.5,
            "min": 0.0,
            "sum": 1.0,
            "num_predictions": 8,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_absolute_relative_error_forward_multiple_batches_with_reset(
    device: str, mode: bool
) -> None:
    device = torch.device(device)
    metric = AbsoluteRelativeError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric.reset()
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device) + 1)
    assert objects_are_equal(
        metric.value(),
        {
            "mean": 0.25,
            "max": 0.5,
            "min": 0.0,
            "sum": 1.0,
            "num_predictions": 4,
        },
    )


def test_absolute_relative_error_value_empty() -> None:
    with pytest.raises(EmptyMetricError, match="ErrorState is empty"):
        AbsoluteRelativeError().value()


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("prefix", ["abs_err_", "abs_err/"])
@pytest.mark.parametrize("suffix", ["_abs_err", "/abs_err"])
def test_absolute_error_value_prefix_suffix(device: str, prefix: str, suffix: str) -> None:
    device = torch.device(device)
    metric = AbsoluteRelativeError().to(device=device)
    metric(torch.ones(2, device=device), torch.ones(2, device=device))
    assert objects_are_equal(
        metric.value(prefix, suffix),
        {
            f"{prefix}mean{suffix}": 0.0,
            f"{prefix}max{suffix}": 0.0,
            f"{prefix}min{suffix}": 0.0,
            f"{prefix}sum{suffix}": 0.0,
            f"{prefix}num_predictions{suffix}": 2,
        },
    )


def test_absolute_relative_error_reset() -> None:
    state = Mock(spec=BaseState)
    metric = AbsoluteRelativeError(state=state)
    metric.reset()
    state.reset.assert_called_once_with()
