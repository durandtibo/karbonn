from unittest.mock import Mock

import pytest
import torch
from coola import objects_are_equal
from coola.utils.tensor import get_available_devices
from minrecord import MinScalarRecord

from karbonn.metric import AbsoluteError, EmptyMetricError
from karbonn.metric.state import BaseState, ErrorState, MeanErrorState

DTYPES = (torch.long, torch.float)
MODES = (True, False)
SIZES = (1, 2)


###################################
#     Tests for AbsoluteError     #
###################################


def test_absolute_error_str() -> None:
    assert str(AbsoluteError()).startswith("AbsoluteError(")


def test_absolute_error_init_state_default() -> None:
    assert isinstance(AbsoluteError().state, ErrorState)


def test_absolute_error_init_state_mean() -> None:
    assert isinstance(AbsoluteError(state=MeanErrorState()).state, MeanErrorState)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_absolute_error_forward_correct(
    device: str, mode: bool, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = AbsoluteError().to(device=device)
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
def test_absolute_error_forward_incorrect(
    device: str, mode: bool, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = AbsoluteError().to(device=device)
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
def test_absolute_error_forward_1d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = AbsoluteError().to(device=device)
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
def test_absolute_error_forward_2d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = AbsoluteError().to(device=device)
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
def test_absolute_error_forward_3d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = AbsoluteError().to(device=device)
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
def test_absolute_error_forward_dtype(
    device: str,
    mode: bool,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = AbsoluteError().to(device=device)
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
def test_absolute_error_forward_state_mean(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = AbsoluteError(state=MeanErrorState()).to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert objects_are_equal(
        metric.value(),
        {
            "mean": 0.0,
            "num_predictions": 6,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_absolute_error_forward_multiple_batches(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = AbsoluteError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device) + 1)
    assert objects_are_equal(
        metric.value(),
        {
            "mean": 0.25,
            "max": 1.0,
            "min": 0.0,
            "sum": 2.0,
            "num_predictions": 8,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_absolute_error_forward_multiple_batches_with_reset(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = AbsoluteError().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric.reset()
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device) + 1)
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


def test_absolute_error_value_empty() -> None:
    with pytest.raises(EmptyMetricError, match="ErrorState is empty"):
        AbsoluteError().value()


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("prefix", ["prefix_", "suffix/"])
@pytest.mark.parametrize("suffix", ["_prefix", "/suffix"])
def test_absolute_error_value_prefix_suffix(device: str, prefix: str, suffix: str) -> None:
    device = torch.device(device)
    metric = AbsoluteError().to(device=device)
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


def test_absolute_error_reset() -> None:
    state = Mock(spec=BaseState)
    metric = AbsoluteError(state=state)
    metric.reset()
    state.reset.assert_called_once_with()


def test_absolute_error_get_records() -> None:
    metric = AbsoluteError()
    assert objects_are_equal(
        metric.get_records(),
        (
            MinScalarRecord(name="mean"),
            MinScalarRecord(name="min"),
            MinScalarRecord(name="max"),
            MinScalarRecord(name="sum"),
        ),
    )


@pytest.mark.parametrize("prefix", ["prefix_", "suffix/"])
@pytest.mark.parametrize("suffix", ["_prefix", "/suffix"])
def test_absolute_error_get_records_prefix_suffix(prefix: str, suffix: str) -> None:
    metric = AbsoluteError()
    assert objects_are_equal(
        metric.get_records(prefix, suffix),
        (
            MinScalarRecord(name=f"{prefix}mean{suffix}"),
            MinScalarRecord(name=f"{prefix}min{suffix}"),
            MinScalarRecord(name=f"{prefix}max{suffix}"),
            MinScalarRecord(name=f"{prefix}sum{suffix}"),
        ),
    )
