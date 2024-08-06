from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from coola import objects_are_equal
from coola.utils.tensor import get_available_devices
from torch.nn import Identity

from karbonn.metric import BinaryAccuracy, CategoricalAccuracy, EmptyMetricError
from karbonn.metric.state import AccuracyState, BaseState, ExtendedAccuracyState
from karbonn.modules import ToBinaryLabel, ToCategoricalLabel

MODES = (True, False)
SIZES = (1, 2)
DTYPES = (torch.long, torch.float)


####################################
#     Tests for BinaryAccuracy     #
####################################


def test_binary_accuracy_str() -> None:
    assert str(BinaryAccuracy()).startswith("BinaryAccuracy(")


def test_binary_accuracy_state_default() -> None:
    assert isinstance(BinaryAccuracy().state, AccuracyState)


def test_binary_accuracy_state_extended() -> None:
    assert isinstance(BinaryAccuracy(state=ExtendedAccuracyState()).state, ExtendedAccuracyState)


def test_binary_accuracy_transform() -> None:
    assert isinstance(BinaryAccuracy(transform=ToBinaryLabel()).transform, ToBinaryLabel)


def test_binary_accuracy_transform_default() -> None:
    assert isinstance(BinaryAccuracy().transform, Identity)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", SIZES)
def test_binary_accuracy_forward_correct(device: str, mode: bool, batch_size: int) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy().to(device=device)
    metric.train(mode)
    metric(torch.ones(batch_size, device=device), torch.ones(batch_size, device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 1.0, "num_predictions": batch_size})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", SIZES)
def test_binary_accuracy_forward_incorrect(device: str, mode: bool, batch_size: int) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy().to(device=device)
    metric.train(mode)
    metric(torch.zeros(batch_size, device=device), torch.ones(batch_size, device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 0.0, "num_predictions": batch_size})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_binary_accuracy_forward_partially_correct(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy().to(device=device)
    metric.train(mode)
    metric(torch.tensor([0, 1, 0, 1], device=device), torch.tensor([1, 1, 0, 0], device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 0.5, "num_predictions": 4})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_binary_accuracy_forward_1d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, device=device), torch.ones(2, device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 1.0, "num_predictions": 2})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_binary_accuracy_forward_1d_and_2d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, device=device), torch.ones(2, 1, device=device))
    assert objects_are_equal(
        metric.value(),
        {"accuracy": 1.0, "num_predictions": 2},
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_binary_accuracy_forward_2d_and_1d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 1, device=device), torch.ones(2, device=device))
    assert objects_are_equal(
        metric.value(),
        {"accuracy": 1.0, "num_predictions": 2},
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_binary_accuracy_forward_2d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert objects_are_equal(
        metric.value(),
        {"accuracy": 1.0, "num_predictions": 6},
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_binary_accuracy_forward_3d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 3, 4, device=device), torch.ones(2, 3, 4, device=device))
    assert objects_are_equal(
        metric.value(),
        {"accuracy": 1.0, "num_predictions": 24},
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype_prediction", [torch.bool, torch.long, torch.float])
@pytest.mark.parametrize("dtype_target", [torch.bool, torch.long, torch.float])
def test_binary_accuracy_forward_dtypes(
    device: str,
    mode: bool,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy().to(device=device)
    metric.train(mode)
    metric(
        torch.ones(2, 3, device=device, dtype=dtype_prediction),
        torch.ones(2, 3, device=device, dtype=dtype_target),
    )
    assert objects_are_equal(
        metric.value(),
        {"accuracy": 1.0, "num_predictions": 6},
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_binary_accuracy_forward_state(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy(state=ExtendedAccuracyState()).to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert objects_are_equal(
        metric.value(),
        {
            "accuracy": 1.0,
            "error": 0.0,
            "num_correct_predictions": 6,
            "num_incorrect_predictions": 0,
            "num_predictions": 6,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_binary_accuracy_forward_threshold_0(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy(transform=ToBinaryLabel()).to(device=device)
    metric.train(mode)
    metric(torch.tensor([-1, 1, -2, 1], device=device), torch.tensor([1, 1, 0, 0], device=device))
    assert objects_are_equal(
        metric.value(),
        {"accuracy": 0.5, "num_predictions": 4},
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_binary_accuracy_forward_multiple_batches(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy().to(device=device)
    metric.train(mode)
    metric(torch.zeros(4, device=device), torch.ones(4, device=device))
    metric(torch.ones(4, device=device), torch.ones(4, device=device))
    assert objects_are_equal(
        metric.value(),
        {"accuracy": 0.5, "num_predictions": 8},
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_binary_accuracy_forward_multiple_batches_with_reset(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy().to(device=device)
    metric.train(mode)
    metric(torch.zeros(4, device=device), torch.ones(4, device=device))
    metric.reset()
    metric(torch.ones(4, device=device), torch.ones(4, device=device))
    assert objects_are_equal(
        metric.value(),
        {"accuracy": 1.0, "num_predictions": 4},
    )


def test_binary_accuracy_value_empty() -> None:
    with pytest.raises(EmptyMetricError, match="AccuracyState is empty"):
        BinaryAccuracy().value()


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("prefix", ["prefix_", "suffix/"])
@pytest.mark.parametrize("suffix", ["_prefix", "/suffix"])
def test_binary_accuracy_value_prefix_suffix(device: str, prefix: str, suffix: str) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy().to(device=device)
    metric(torch.ones(2, device=device), torch.ones(2, device=device))
    assert objects_are_equal(
        metric.value(prefix, suffix),
        {f"{prefix}accuracy{suffix}": 1.0, f"{prefix}num_predictions{suffix}": 2},
    )


def test_binary_accuracy_reset() -> None:
    state = Mock(spec=BaseState)
    metric = BinaryAccuracy(state=state)
    metric.reset()
    state.reset.assert_called_once_with()


#########################################
#     Tests for CategoricalAccuracy     #
#########################################


def test_categorical_accuracy_str() -> None:
    assert str(CategoricalAccuracy()).startswith("CategoricalAccuracy(")


def test_categorical_accuracy_state_default() -> None:
    assert isinstance(CategoricalAccuracy().state, AccuracyState)


def test_categorical_accuracy_state_extended() -> None:
    assert isinstance(
        CategoricalAccuracy(state=ExtendedAccuracyState()).state, ExtendedAccuracyState
    )


def test_categorical_accuracy_transform() -> None:
    assert isinstance(
        CategoricalAccuracy(transform=ToCategoricalLabel()).transform, ToCategoricalLabel
    )


def test_categorical_accuracy_transform_default() -> None:
    assert isinstance(CategoricalAccuracy().transform, Identity)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", SIZES)
def test_categorical_accuracy_forward_correct(device: str, mode: bool, batch_size: int) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy().to(device=device)
    metric.train(mode)
    metric(torch.arange(batch_size, device=device), torch.arange(batch_size, device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 1.0, "num_predictions": batch_size})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", SIZES)
def test_categorical_accuracy_forward_incorrect(device: str, mode: bool, batch_size: int) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy().to(device=device)
    metric.train(mode)
    metric(torch.zeros(batch_size, device=device), torch.ones(batch_size, device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 0.0, "num_predictions": batch_size})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_accuracy_forward_partially_correct(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy().to(device=device)
    metric.train(mode)
    metric(torch.tensor([0.0, 1.0], device=device), torch.zeros(2, device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 0.5, "num_predictions": 2})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_accuracy_forward_prediction_1d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy().to(device=device)
    metric.train(mode)
    metric(torch.zeros(2, device=device), torch.zeros(2, device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 1.0, "num_predictions": 2})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_accuracy_forward_prediction_1d_target_2d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy().to(device=device)
    metric.train(mode)
    metric(torch.zeros(2, device=device), torch.zeros(2, 1, device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 1.0, "num_predictions": 2})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_accuracy_forward_prediction_2d_target_1d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy().to(device=device)
    metric.train(mode)
    metric(torch.zeros(2, 1, device=device), torch.zeros(2, device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 1.0, "num_predictions": 2})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_accuracy_forward_prediction_2d_target_2d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy().to(device=device)
    metric.train(mode)
    metric(torch.zeros(2, 1, device=device), torch.zeros(2, 1, device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 1.0, "num_predictions": 2})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_accuracy_forward_prediction_3d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy().to(device=device)
    metric.train(mode)
    metric(torch.zeros(2, 3, 4, device=device), torch.zeros(2, 3, 4, device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 1.0, "num_predictions": 24})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype_prediction", DTYPES)
@pytest.mark.parametrize("dtype_target", DTYPES)
def test_categorical_accuracy_forward_dtypes(
    device: str,
    mode: bool,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy().to(device=device)
    metric.train(mode)
    metric(
        torch.arange(2, device=device, dtype=dtype_prediction),
        torch.arange(2, device=device, dtype=dtype_target),
    )
    assert objects_are_equal(metric.value(), {"accuracy": 1.0, "num_predictions": 2})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_accuracy_forward_state(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy(state=ExtendedAccuracyState()).to(device=device)
    metric.train(mode)
    metric(torch.tensor([0.0, 1.0], device=device), torch.zeros(2, device=device))
    assert objects_are_equal(
        metric.value(),
        {
            "accuracy": 0.5,
            "error": 0.5,
            "num_correct_predictions": 1,
            "num_incorrect_predictions": 1,
            "num_predictions": 2,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_accuracy_forward_multiple_batches(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy().to(device=device)
    metric.train(mode)
    metric(torch.zeros(2, device=device), torch.zeros(2, device=device))
    metric(torch.zeros(2, device=device), torch.tensor([0.0, 1.0], device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 0.75, "num_predictions": 4})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_accuracy_forward_multiple_batches_with_reset(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy().to(device=device)
    metric.train(mode)
    metric(torch.zeros(2, device=device), torch.zeros(2, device=device))
    metric.reset()
    metric(torch.tensor([0.0, 1.0], device=device), torch.tensor([0.0, 1.0], device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 1.0, "num_predictions": 2})


def test_categorical_accuracy_value_empty() -> None:
    with pytest.raises(EmptyMetricError, match="AccuracyState is empty"):
        CategoricalAccuracy().value()


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("prefix", ["prefix_", "suffix/"])
@pytest.mark.parametrize("suffix", ["_prefix", "/suffix"])
def test_categorical_accuracy_value_prefix_suffix(device: str, prefix: str, suffix: str) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy().to(device=device)
    metric(torch.ones(2, device=device), torch.ones(2, device=device))
    assert objects_are_equal(
        metric.value(prefix, suffix),
        {f"{prefix}accuracy{suffix}": 1.0, f"{prefix}num_predictions{suffix}": 2},
    )


def test_categorical_accuracy_reset() -> None:
    state = Mock(spec=BaseState)
    metric = CategoricalAccuracy(state=state)
    metric.reset()
    state.reset.assert_called_once_with()
