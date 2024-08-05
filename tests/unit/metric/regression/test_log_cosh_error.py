from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices

from karbonn.metric import EmptyMetricError, LogCoshError
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
#     Tests for LogCoshError     #
##################################


def test_log_cosh_error_str() -> None:
    assert str(LogCoshError()).startswith("LogCoshError(")


def test_log_cosh_error_init_scale_default() -> None:
    assert LogCoshError()._scale == 1.0


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_log_cosh_error_init_scale(scale: float) -> None:
    assert LogCoshError(scale=scale)._scale == scale


def test_log_cosh_error_init_scale_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect scale"):
        LogCoshError(scale=0.0)


def test_log_cosh_error_init_state_default() -> None:
    assert isinstance(LogCoshError().state, ErrorState)


def test_log_cosh_error_init_state_mean() -> None:
    assert isinstance(LogCoshError(state=MeanErrorState()).state, MeanErrorState)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_log_cosh_error_forward_correct(
    device: str, mode: bool, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = LogCoshError().to(device=device)
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
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_log_cosh_error_forward_incorrect(
    device: str, mode: bool, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = LogCoshError().to(device=device)
    metric.train(mode)
    metric(
        torch.zeros(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert objects_are_allclose(
        metric.value(),
        {
            "mean": 0.4337808304830271,
            "max": 0.4337808304830271,
            "min": 0.4337808304830271,
            "sum": 0.4337808304830271 * batch_size * feature_size,
            "num_predictions": batch_size * feature_size,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_log_cosh_error_forward_partially_correct(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = LogCoshError().to(device=device)
    metric.train(mode)
    metric(torch.eye(2, device=device), torch.ones(2, 2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            "mean": 0.21689041524151356,
            "max": 0.4337808304830271,
            "min": 0.0,
            "sum": 0.8675616609660542,
            "num_predictions": 4,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_log_cosh_error_forward_1d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = LogCoshError().to(device=device)
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
def test_log_cosh_error_forward_2d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = LogCoshError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
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
def test_log_cosh_error_forward_3d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = LogCoshError().to(device=device)
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
def test_log_cosh_error_forward_scale_2(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = LogCoshError(scale=2.0).to(device=device)
    metric.train(mode)
    metric(torch.eye(2, device=device), torch.ones(2, 2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            "mean": 0.06005725347913873,
            "max": 0.12011450695827745,
            "min": 0.0,
            "sum": 0.2402290139165549,
            "num_predictions": 4,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype_prediction", DTYPES)
@pytest.mark.parametrize("dtype_target", DTYPES)
def test_log_cosh_error_forward_dtypes(
    device: str,
    mode: bool,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = LogCoshError().to(device=device)
    metric.train(mode)
    metric(
        torch.ones(2, 2, device=device, dtype=dtype_prediction),
        torch.ones(2, 2, device=device, dtype=dtype_target),
    )
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
def test_log_cosh_error_forward_state(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = LogCoshError(state=ExtendedErrorState()).to(device=device)
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
def test_log_cosh_error_forward_multiple_batches(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = LogCoshError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric(torch.eye(2, device=device), torch.ones(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            "mean": 0.10844520762075678,
            "max": 0.4337808304830271,
            "min": 0.0,
            "sum": 0.8675616609660542,
            "num_predictions": 8,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_log_cosh_error_forward_multiple_batches_with_reset(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = LogCoshError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric.reset()
    metric(torch.eye(2, device=device), torch.ones(2, 2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            "mean": 0.21689041524151356,
            "max": 0.4337808304830271,
            "min": 0.0,
            "sum": 0.8675616609660542,
            "num_predictions": 4,
        },
    )


def test_log_cosh_error_value_empty() -> None:
    with pytest.raises(EmptyMetricError, match="ErrorState is empty"):
        LogCoshError().value()


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("prefix", ["prefix_", "suffix/"])
@pytest.mark.parametrize("suffix", ["_prefix", "/suffix"])
def test_log_cosh_error_value_prefix_suffix(device: str, prefix: str, suffix: str) -> None:
    device = torch.device(device)
    metric = LogCoshError().to(device=device)
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


def test_log_cosh_error_reset() -> None:
    state = Mock(spec=BaseState)
    metric = LogCoshError(state=state)
    metric.reset()
    state.reset.assert_called_once_with()
