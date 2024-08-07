from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
import torch
from coola import objects_are_equal
from coola.utils.tensor import get_available_devices
from minrecord import MaxScalarRecord
from torch.nn import Identity

from karbonn.metric import (
    BinaryAccuracy,
    CategoricalAccuracy,
    EmptyMetricError,
    TopKAccuracy,
)
from karbonn.metric.state import AccuracyState, BaseState, ExtendedAccuracyState
from karbonn.modules import ToBinaryLabel, ToCategoricalLabel

if TYPE_CHECKING:
    from collections.abc import Sequence

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


def test_binary_accuracy_get_records() -> None:
    metric = BinaryAccuracy()
    assert objects_are_equal(
        metric.get_records(),
        (MaxScalarRecord(name="accuracy"),),
    )


@pytest.mark.parametrize("prefix", ["prefix_", "suffix/"])
@pytest.mark.parametrize("suffix", ["_prefix", "/suffix"])
def test_binary_accuracy_get_records_prefix_suffix(prefix: str, suffix: str) -> None:
    metric = BinaryAccuracy()
    assert objects_are_equal(
        metric.get_records(prefix, suffix),
        (MaxScalarRecord(name=f"{prefix}accuracy{suffix}"),),
    )


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


def test_categorical_accuracy_get_records() -> None:
    metric = CategoricalAccuracy()
    assert objects_are_equal(
        metric.get_records(),
        (MaxScalarRecord(name="accuracy"),),
    )


@pytest.mark.parametrize("prefix", ["prefix_", "suffix/"])
@pytest.mark.parametrize("suffix", ["_prefix", "/suffix"])
def test_categorical_accuracy_get_records_prefix_suffix(prefix: str, suffix: str) -> None:
    metric = CategoricalAccuracy()
    assert objects_are_equal(
        metric.get_records(prefix, suffix),
        (MaxScalarRecord(name=f"{prefix}accuracy{suffix}"),),
    )


##################################
#     Tests for TopKAccuracy     #
##################################


def test_top_k_accuracy_str() -> None:
    assert str(TopKAccuracy()).startswith("TopKAccuracy(")


@pytest.mark.parametrize(
    ("topk", "tuple_topk"),
    [
        ((1,), (1,)),
        ((1, 5), (1, 5)),
        ([1], (1,)),
        ([1, 5], (1, 5)),
    ],
)
def test_top_k_accuracy_tolerances(topk: Sequence[int], tuple_topk: tuple[int, ...]) -> None:
    assert TopKAccuracy(topk=topk).topk == tuple_topk


def test_top_k_accuracy_state_default() -> None:
    metric = TopKAccuracy(topk=(1, 5))
    assert objects_are_equal(metric._states, {1: AccuracyState(), 5: AccuracyState()})


def test_top_k_accuracy_state_extended() -> None:
    metric = TopKAccuracy(topk=(1, 5), state=ExtendedAccuracyState())
    assert objects_are_equal(
        metric._states, {1: ExtendedAccuracyState(), 5: ExtendedAccuracyState()}
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", SIZES)
def test_top_k_accuracy_forward_top1_correct(device: str, mode: bool, batch_size: int) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(topk=(1,)).to(device=device)
    metric.train(mode)
    metric(
        prediction=torch.eye(batch_size, device=device),
        target=torch.arange(batch_size, device=device),
    )
    assert objects_are_equal(
        metric.value(), {"top_1_accuracy": 1.0, "top_1_num_predictions": batch_size}
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_top_1_5_correct(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(topk=(1, 5)).to(device=device)
    metric.train(mode)
    metric(prediction=torch.eye(10, device=device), target=torch.arange(10, device=device))
    assert objects_are_equal(
        metric.value(),
        {
            "top_1_accuracy": 1.0,
            "top_1_num_predictions": 10,
            "top_5_accuracy": 1.0,
            "top_5_num_predictions": 10,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_top_1_partially_correct(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(topk=(1,)).to(device=device)
    metric.train(mode)
    metric(
        prediction=torch.tensor([[0, 2, 1], [2, 1, 0]], device=device),
        target=torch.tensor([1, 2], device=device),
    )
    assert objects_are_equal(metric.value(), {"top_1_accuracy": 0.5, "top_1_num_predictions": 2})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_top_1_2_3_partially_correct(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(topk=(1, 2, 3)).to(device=device)
    metric.train(mode)
    metric(
        prediction=torch.tensor([[0, 2, 1], [2, 1, 0]], device=device),
        target=torch.tensor([1, 2], device=device),
    )
    assert objects_are_equal(
        metric.value(),
        {
            "top_1_accuracy": 0.5,
            "top_1_num_predictions": 2,
            "top_2_accuracy": 0.5,
            "top_2_num_predictions": 2,
            "top_3_accuracy": 1.0,
            "top_3_num_predictions": 2,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_top_1_incorrect(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(topk=(1,)).to(device=device)
    metric.train(mode)
    metric(
        prediction=torch.tensor([[0, 1, 2], [2, 1, 0]], device=device),
        target=torch.tensor([1, 2], device=device),
    )
    assert objects_are_equal(metric.value(), {"top_1_accuracy": 0.0, "top_1_num_predictions": 2})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_top_1_2_3_incorrect(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(topk=(1, 2, 3)).to(device=device)
    metric.train(mode)
    metric(
        prediction=torch.tensor([[0, 1, 2], [2, 1, 0]], device=device),
        target=torch.tensor([1, 2], device=device),
    )
    assert objects_are_equal(
        metric.value(),
        {
            "top_1_accuracy": 0.0,
            "top_1_num_predictions": 2,
            "top_2_accuracy": 0.5,
            "top_2_num_predictions": 2,
            "top_3_accuracy": 1.0,
            "top_3_num_predictions": 2,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_prediction_2d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(topk=(1,)).to(device=device)
    metric.train(mode)
    prediction = torch.rand(2, 3, device=device)
    prediction[..., 0] = 2
    metric(prediction=prediction, target=torch.zeros(2, device=device))
    assert objects_are_equal(metric.value(), {"top_1_accuracy": 1.0, "top_1_num_predictions": 2})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_prediction_2d_target_2d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(topk=(1,)).to(device=device)
    metric.train(mode)
    prediction = torch.rand(2, 3, device=device)
    prediction[..., 0] = 2
    metric(prediction=prediction, target=torch.zeros(2, 1, device=device))
    assert objects_are_equal(metric.value(), {"top_1_accuracy": 1.0, "top_1_num_predictions": 2})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_prediction_3d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(topk=(1,)).to(device=device)
    metric.train(mode)
    prediction = torch.rand(2, 3, 4, device=device)
    prediction[..., 0] = 2
    metric(prediction=prediction, target=torch.zeros(2, 3, device=device))
    assert objects_are_equal(metric.value(), {"top_1_accuracy": 1.0, "top_1_num_predictions": 6})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_prediction_4d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(topk=(1,)).to(device=device)
    metric.train(mode)
    prediction = torch.rand(2, 3, 4, 5, device=device)
    prediction[..., 0] = 2
    metric(prediction=prediction, target=torch.zeros(2, 3, 4, device=device))
    assert objects_are_equal(metric.value(), {"top_1_accuracy": 1.0, "top_1_num_predictions": 24})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype_prediction", DTYPES)
@pytest.mark.parametrize("dtype_target", DTYPES)
def test_top_k_accuracy_forward_dtypes(
    device: str,
    mode: bool,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(topk=(1,)).to(device=device)
    metric.train(mode)
    metric(
        torch.eye(4, device=device, dtype=dtype_prediction),
        torch.arange(4, device=device, dtype=dtype_target),
    )
    assert objects_are_equal(metric.value(), {"top_1_accuracy": 1.0, "top_1_num_predictions": 4})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_state(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(topk=(1, 2, 3), state=ExtendedAccuracyState()).to(device=device)
    metric.train(mode)
    metric(
        prediction=torch.tensor([[0, 2, 1], [2, 1, 0]], device=device),
        target=torch.tensor([1, 2], device=device),
    )
    assert objects_are_equal(
        metric.value(),
        {
            "top_1_accuracy": 0.5,
            "top_1_error": 0.5,
            "top_1_num_correct_predictions": 1,
            "top_1_num_incorrect_predictions": 1,
            "top_1_num_predictions": 2,
            "top_2_accuracy": 0.5,
            "top_2_error": 0.5,
            "top_2_num_correct_predictions": 1,
            "top_2_num_incorrect_predictions": 1,
            "top_2_num_predictions": 2,
            "top_3_accuracy": 1.0,
            "top_3_error": 0.0,
            "top_3_num_correct_predictions": 2,
            "top_3_num_incorrect_predictions": 0,
            "top_3_num_predictions": 2,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_top_1_multiple_batches(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(topk=(1,)).to(device=device)
    metric.train(mode)
    metric(
        prediction=torch.tensor([[0, 2, 1], [2, 1, 0]], device=device),
        target=torch.tensor([1, 0], device=device),
    )
    metric(
        prediction=torch.tensor([[0, 2, 1], [2, 1, 0]], device=device),
        target=torch.tensor([1, 2], device=device),
    )
    assert objects_are_equal(
        metric.value(),
        {"top_1_accuracy": 0.75, "top_1_num_predictions": 4},
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_top_1_multiple_batches_with_reset(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(topk=(1,)).to(device=device)
    metric.train(mode)
    metric(
        prediction=torch.tensor([[0, 2, 1], [2, 1, 0]], device=device),
        target=torch.tensor([1, 0], device=device),
    )
    metric.reset()
    metric(
        prediction=torch.tensor([[0, 2, 1], [2, 1, 0]], device=device),
        target=torch.tensor([1, 2], device=device),
    )
    assert objects_are_equal(metric.value(), {"top_1_accuracy": 0.5, "top_1_num_predictions": 2})


def test_top_k_accuracy_value_empty() -> None:
    with pytest.raises(EmptyMetricError, match="AccuracyState is empty"):
        TopKAccuracy().value()


def test_top_k_accuracy_reset() -> None:
    metric = TopKAccuracy(topk=(1, 3))
    metric(prediction=torch.eye(4), target=torch.ones(4))
    metric.reset()
    assert metric._states[1].num_predictions == 0
    assert metric._states[3].num_predictions == 0


def test_top_k_accuracy_get_records() -> None:
    metric = TopKAccuracy(topk=(1, 3))
    assert objects_are_equal(
        metric.get_records(),
        (
            MaxScalarRecord(name="top_1_accuracy"),
            MaxScalarRecord(name="top_3_accuracy"),
        ),
    )


@pytest.mark.parametrize("prefix", ["prefix_", "suffix/"])
@pytest.mark.parametrize("suffix", ["_prefix", "/suffix"])
def test_top_k_accuracy_get_records_prefix_suffix(prefix: str, suffix: str) -> None:
    metric = TopKAccuracy(topk=(1, 3))
    assert objects_are_equal(
        metric.get_records(prefix, suffix),
        (
            MaxScalarRecord(name=f"{prefix}top_1_accuracy{suffix}"),
            MaxScalarRecord(name=f"{prefix}top_3_accuracy{suffix}"),
        ),
    )
