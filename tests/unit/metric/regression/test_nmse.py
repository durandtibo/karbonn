from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from coola import objects_are_equal
from coola.utils.tensor import get_available_devices
from minrecord import MinScalarRecord

from karbonn.metric import EmptyMetricError, NormalizedMeanSquaredError
from karbonn.metric.state import BaseState, NormalizedMeanSquaredErrorState

MODES = (True, False)
SIZES = (1, 2)
DTYPES = (torch.long, torch.float)


################################################
#     Tests for NormalizedMeanSquaredError     #
################################################


def test_normalized_mean_squared_error_str() -> None:
    assert str(NormalizedMeanSquaredError()).startswith("NormalizedMeanSquaredError(")


def test_absolute_error_init_state_default() -> None:
    assert isinstance(NormalizedMeanSquaredError().state, NormalizedMeanSquaredErrorState)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_normalized_mean_squared_error_forward_correct(
    device: str, mode: bool, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = NormalizedMeanSquaredError().to(device=device)
    metric.train(mode)
    metric(
        torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert objects_are_equal(
        metric.value(),
        {
            "mean": 0.0,
            "count": batch_size * feature_size,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_normalized_mean_squared_error_forward_incorrect(
    device: str, mode: bool, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = NormalizedMeanSquaredError().to(device=device)
    metric.train(mode)
    metric(
        -torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert objects_are_equal(
        metric.value(),
        {
            "mean": 4.0,
            "count": batch_size * feature_size,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_normalized_mean_squared_error_forward_partially_correct(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = NormalizedMeanSquaredError().to(device=device)
    metric.train(mode)
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    assert objects_are_equal(metric.value(), {"mean": 4.0, "count": 4})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_normalized_mean_squared_error_forward_1d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = NormalizedMeanSquaredError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, device=device), torch.ones(2, device=device))
    assert objects_are_equal(metric.value(), {"mean": 0.0, "count": 2})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_normalized_mean_squared_error_forward_2d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = NormalizedMeanSquaredError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert objects_are_equal(metric.value(), {"mean": 0.0, "count": 6})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_normalized_mean_squared_error_forward_3d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = NormalizedMeanSquaredError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 3, 4, device=device), torch.ones(2, 3, 4, device=device))
    assert objects_are_equal(metric.value(), {"mean": 0.0, "count": 24})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype_prediction", DTYPES)
@pytest.mark.parametrize("dtype_target", DTYPES)
def test_normalized_mean_squared_error_forward_dtypes(
    device: str,
    mode: bool,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = NormalizedMeanSquaredError().to(device=device)
    metric.train(mode)
    metric(
        torch.ones(2, 3, device=device, dtype=dtype_prediction),
        torch.ones(2, 3, device=device, dtype=dtype_target),
    )
    assert objects_are_equal(metric.value(), {"mean": 0.0, "count": 6})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_normalized_mean_squared_error_forward_multiple_batches(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = NormalizedMeanSquaredError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric(torch.ones(2, 2, device=device), -torch.ones(2, 2, device=device))
    assert objects_are_equal(metric.value(), {"mean": 2.0, "count": 8})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_normalized_mean_squared_error_forward_multiple_batches_with_reset(
    device: str, mode: bool
) -> None:
    device = torch.device(device)
    metric = NormalizedMeanSquaredError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric.reset()
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    assert objects_are_equal(metric.value(), {"mean": 4.0, "count": 4})


def test_normalized_mean_squared_error_value_empty() -> None:
    with pytest.raises(EmptyMetricError, match="NormalizedMeanSquaredErrorState is empty"):
        NormalizedMeanSquaredError().value()


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("prefix", ["prefix_", "suffix/"])
@pytest.mark.parametrize("suffix", ["_prefix", "/suffix"])
def test_normalized_mean_squared_error_value_prefix_suffix(
    device: str, prefix: str, suffix: str
) -> None:
    device = torch.device(device)
    metric = NormalizedMeanSquaredError().to(device=device)
    metric(torch.ones(2, device=device), torch.ones(2, device=device))
    assert objects_are_equal(
        metric.value(prefix, suffix),
        {
            f"{prefix}mean{suffix}": 0.0,
            f"{prefix}count{suffix}": 2,
        },
    )


def test_normalized_mean_squared_error_reset() -> None:
    state = Mock(spec=BaseState)
    metric = NormalizedMeanSquaredError(state=state)
    metric.reset()
    state.reset.assert_called_once_with()


def test_normalized_mean_squared_error_get_records() -> None:
    metric = NormalizedMeanSquaredError()
    assert objects_are_equal(
        metric.get_records(),
        (MinScalarRecord(name="mean"),),
    )


@pytest.mark.parametrize("prefix", ["prefix_", "suffix/"])
@pytest.mark.parametrize("suffix", ["_prefix", "/suffix"])
def test_normalized_mean_squared_error_get_records_prefix_suffix(prefix: str, suffix: str) -> None:
    metric = NormalizedMeanSquaredError()
    assert objects_are_equal(
        metric.get_records(prefix, suffix),
        (MinScalarRecord(name=f"{prefix}mean{suffix}"),),
    )
