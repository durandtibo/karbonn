from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices
from minrecord import MaxScalarRecord
from torch.nn import Identity

from karbonn.metric import Accuracy, EmptyMetricError, TopKAccuracy
from karbonn.metric.state import AccuracyState, BaseState, ExtendedAccuracyState
from karbonn.modules import ToBinaryLabel, ToCategoricalLabel
from karbonn.testing import sklearn_available
from karbonn.utils.imports import is_sklearn_available

if TYPE_CHECKING:
    from collections.abc import Sequence

if is_sklearn_available():
    from sklearn import metrics


MODES = (True, False)
SIZES = (1, 2)
DTYPES = (torch.long, torch.float)


##############################
#     Tests for Accuracy     #
##############################


def test_accuracy_str() -> None:
    assert str(Accuracy()).startswith("Accuracy(")


def test_accuracy_state_default() -> None:
    assert isinstance(Accuracy().state, AccuracyState)


def test_accuracy_state_extended() -> None:
    assert isinstance(Accuracy(state=ExtendedAccuracyState()).state, ExtendedAccuracyState)


def test_accuracy_transform() -> None:
    assert isinstance(Accuracy(transform=ToCategoricalLabel()).transform, ToCategoricalLabel)


def test_accuracy_transform_default() -> None:
    assert isinstance(Accuracy().transform, Identity)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", SIZES)
def test_accuracy_forward_correct(device: str, mode: bool, batch_size: int) -> None:
    device = torch.device(device)
    metric = Accuracy().to(device=device)
    metric.train(mode)
    metric(torch.arange(batch_size, device=device), torch.arange(batch_size, device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 1.0, "count": batch_size})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", SIZES)
def test_accuracy_forward_incorrect(device: str, mode: bool, batch_size: int) -> None:
    device = torch.device(device)
    metric = Accuracy().to(device=device)
    metric.train(mode)
    metric(torch.zeros(batch_size, device=device), torch.ones(batch_size, device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 0.0, "count": batch_size})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_accuracy_forward_partially_correct(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = Accuracy().to(device=device)
    metric.train(mode)
    metric(torch.tensor([0.0, 1.0], device=device), torch.zeros(2, device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 0.5, "count": 2})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_accuracy_forward_prediction_1d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = Accuracy().to(device=device)
    metric.train(mode)
    metric(torch.zeros(2, device=device), torch.zeros(2, device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 1.0, "count": 2})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_accuracy_forward_prediction_1d_target_2d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = Accuracy().to(device=device)
    metric.train(mode)
    metric(torch.zeros(2, device=device), torch.zeros(2, 1, device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 1.0, "count": 2})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_accuracy_forward_prediction_2d_target_1d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = Accuracy().to(device=device)
    metric.train(mode)
    metric(torch.zeros(2, 1, device=device), torch.zeros(2, device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 1.0, "count": 2})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_accuracy_forward_prediction_2d_target_2d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = Accuracy().to(device=device)
    metric.train(mode)
    metric(torch.zeros(2, 1, device=device), torch.zeros(2, 1, device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 1.0, "count": 2})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_accuracy_forward_prediction_3d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = Accuracy().to(device=device)
    metric.train(mode)
    metric(torch.zeros(2, 3, 4, device=device), torch.zeros(2, 3, 4, device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 1.0, "count": 24})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_accuracy_forward_transform_binary(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = Accuracy(transform=ToBinaryLabel()).to(device=device)
    metric.train(mode)
    metric(torch.tensor([-1, 1, -2, 1], device=device), torch.tensor([1, 1, 0, 0], device=device))
    assert objects_are_equal(
        metric.value(),
        {"accuracy": 0.5, "count": 4},
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_accuracy_forward_transform_categorical(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = Accuracy(transform=ToCategoricalLabel()).to(device=device)
    metric.train(mode)
    metric(
        torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 3.0, 2.0, 2.0]], device=device),
        torch.tensor([3, 0], device=device),
    )
    assert objects_are_equal(metric.value(), {"accuracy": 1.0, "count": 2})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype_prediction", DTYPES)
@pytest.mark.parametrize("dtype_target", DTYPES)
def test_accuracy_forward_dtypes(
    device: str,
    mode: bool,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = Accuracy().to(device=device)
    metric.train(mode)
    metric(
        torch.arange(2, device=device, dtype=dtype_prediction),
        torch.arange(2, device=device, dtype=dtype_target),
    )
    assert objects_are_equal(metric.value(), {"accuracy": 1.0, "count": 2})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_accuracy_forward_state(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = Accuracy(state=ExtendedAccuracyState()).to(device=device)
    metric.train(mode)
    metric(torch.tensor([0.0, 1.0], device=device), torch.zeros(2, device=device))
    assert objects_are_equal(
        metric.value(),
        {
            "accuracy": 0.5,
            "error": 0.5,
            "count_correct": 1,
            "count_incorrect": 1,
            "count": 2,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_accuracy_forward_multiple_batches(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = Accuracy().to(device=device)
    metric.train(mode)
    metric(torch.zeros(2, device=device), torch.zeros(2, device=device))
    metric(torch.zeros(2, device=device), torch.tensor([0.0, 1.0], device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 0.75, "count": 4})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_accuracy_forward_multiple_batches_with_reset(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = Accuracy().to(device=device)
    metric.train(mode)
    metric(torch.zeros(2, device=device), torch.zeros(2, device=device))
    metric.reset()
    metric(torch.tensor([0.0, 1.0], device=device), torch.tensor([0.0, 1.0], device=device))
    assert objects_are_equal(metric.value(), {"accuracy": 1.0, "count": 2})


def test_accuracy_value_empty() -> None:
    with pytest.raises(EmptyMetricError, match="AccuracyState is empty"):
        Accuracy().value()


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("prefix", ["prefix_", "suffix/"])
@pytest.mark.parametrize("suffix", ["_prefix", "/suffix"])
def test_accuracy_value_prefix_suffix(device: str, prefix: str, suffix: str) -> None:
    device = torch.device(device)
    metric = Accuracy().to(device=device)
    metric(torch.ones(2, device=device), torch.ones(2, device=device))
    assert objects_are_equal(
        metric.value(prefix, suffix),
        {f"{prefix}accuracy{suffix}": 1.0, f"{prefix}count{suffix}": 2},
    )


def test_accuracy_reset() -> None:
    state = Mock(spec=BaseState)
    metric = Accuracy(state=state)
    metric.reset()
    state.reset.assert_called_once_with()


def test_accuracy_get_records() -> None:
    metric = Accuracy()
    assert objects_are_equal(
        metric.get_records(),
        (MaxScalarRecord(name="accuracy"),),
    )


@pytest.mark.parametrize("prefix", ["prefix_", "suffix/"])
@pytest.mark.parametrize("suffix", ["_prefix", "/suffix"])
def test_accuracy_get_records_prefix_suffix(prefix: str, suffix: str) -> None:
    metric = Accuracy()
    assert objects_are_equal(
        metric.get_records(prefix, suffix),
        (MaxScalarRecord(name=f"{prefix}accuracy{suffix}"),),
    )


@sklearn_available
def test_accuracy_value_sklearn() -> None:
    metric = Accuracy()
    prediction = torch.randint(0, 10, size=(100,))
    target = torch.randint(0, 10, size=(100,))
    metric(prediction=prediction, target=target)
    assert objects_are_allclose(
        metric.value(),
        {
            "accuracy": metrics.accuracy_score(y_true=target.numpy(), y_pred=prediction.numpy()),
            "count": 100,
        },
    )


@sklearn_available
def test_accuracy_value_sklearn_extended() -> None:
    metric = Accuracy(state=ExtendedAccuracyState())
    prediction = torch.randint(0, 10, size=(100,))
    target = torch.randint(0, 10, size=(100,))
    metric(prediction=prediction, target=target)
    assert objects_are_allclose(
        metric.value(),
        {
            "accuracy": metrics.accuracy_score(y_true=target.numpy(), y_pred=prediction.numpy()),
            "count": 100,
            "count_correct": int(
                metrics.accuracy_score(
                    y_true=target.numpy(), y_pred=prediction.numpy(), normalize=False
                )
            ),
            "count_incorrect": 100
            - int(
                metrics.accuracy_score(
                    y_true=target.numpy(), y_pred=prediction.numpy(), normalize=False
                )
            ),
            "error": 1.0 - metrics.accuracy_score(y_true=target.numpy(), y_pred=prediction.numpy()),
        },
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
    assert objects_are_equal(metric.value(), {"top_1_accuracy": 1.0, "top_1_count": batch_size})


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
            "top_1_count": 10,
            "top_5_accuracy": 1.0,
            "top_5_count": 10,
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
    assert objects_are_equal(metric.value(), {"top_1_accuracy": 0.5, "top_1_count": 2})


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
            "top_1_count": 2,
            "top_2_accuracy": 0.5,
            "top_2_count": 2,
            "top_3_accuracy": 1.0,
            "top_3_count": 2,
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
    assert objects_are_equal(metric.value(), {"top_1_accuracy": 0.0, "top_1_count": 2})


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
            "top_1_count": 2,
            "top_2_accuracy": 0.5,
            "top_2_count": 2,
            "top_3_accuracy": 1.0,
            "top_3_count": 2,
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
    assert objects_are_equal(metric.value(), {"top_1_accuracy": 1.0, "top_1_count": 2})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_prediction_2d_target_2d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(topk=(1,)).to(device=device)
    metric.train(mode)
    prediction = torch.rand(2, 3, device=device)
    prediction[..., 0] = 2
    metric(prediction=prediction, target=torch.zeros(2, 1, device=device))
    assert objects_are_equal(metric.value(), {"top_1_accuracy": 1.0, "top_1_count": 2})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_prediction_3d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(topk=(1,)).to(device=device)
    metric.train(mode)
    prediction = torch.rand(2, 3, 4, device=device)
    prediction[..., 0] = 2
    metric(prediction=prediction, target=torch.zeros(2, 3, device=device))
    assert objects_are_equal(metric.value(), {"top_1_accuracy": 1.0, "top_1_count": 6})


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_prediction_4d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(topk=(1,)).to(device=device)
    metric.train(mode)
    prediction = torch.rand(2, 3, 4, 5, device=device)
    prediction[..., 0] = 2
    metric(prediction=prediction, target=torch.zeros(2, 3, 4, device=device))
    assert objects_are_equal(metric.value(), {"top_1_accuracy": 1.0, "top_1_count": 24})


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
    assert objects_are_equal(metric.value(), {"top_1_accuracy": 1.0, "top_1_count": 4})


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
            "top_1_count_correct": 1,
            "top_1_count_incorrect": 1,
            "top_1_count": 2,
            "top_2_accuracy": 0.5,
            "top_2_error": 0.5,
            "top_2_count_correct": 1,
            "top_2_count_incorrect": 1,
            "top_2_count": 2,
            "top_3_accuracy": 1.0,
            "top_3_error": 0.0,
            "top_3_count_correct": 2,
            "top_3_count_incorrect": 0,
            "top_3_count": 2,
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
        {"top_1_accuracy": 0.75, "top_1_count": 4},
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
    assert objects_are_equal(metric.value(), {"top_1_accuracy": 0.5, "top_1_count": 2})


def test_top_k_accuracy_value_empty() -> None:
    with pytest.raises(EmptyMetricError, match="AccuracyState is empty"):
        TopKAccuracy().value()


def test_top_k_accuracy_reset() -> None:
    metric = TopKAccuracy(topk=(1, 3))
    metric(prediction=torch.eye(4), target=torch.ones(4))
    metric.reset()
    assert metric._states[1].count == 0
    assert metric._states[3].count == 0


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


@sklearn_available
def test_top_k_accuracy_value_sklearn() -> None:
    metric = TopKAccuracy(topk=(1, 3))
    prediction = torch.randn(size=(100, 10))
    target = torch.randint(0, 10, size=(100,))
    metric(prediction=prediction, target=target)
    assert objects_are_allclose(
        metric.value(),
        {
            "top_1_accuracy": float(
                metrics.top_k_accuracy_score(y_true=target.numpy(), y_score=prediction.numpy(), k=1)
            ),
            "top_1_count": 100,
            "top_3_accuracy": float(
                metrics.top_k_accuracy_score(y_true=target.numpy(), y_score=prediction.numpy(), k=3)
            ),
            "top_3_count": 100,
        },
    )


@sklearn_available
def test_top_k_accuracy_value_sklearn_extended() -> None:
    metric = TopKAccuracy(topk=(1, 3), state=ExtendedAccuracyState())
    prediction = torch.randn(size=(100, 10))
    target = torch.randint(0, 10, size=(100,))
    metric(prediction=prediction, target=target)
    assert objects_are_allclose(
        metric.value(),
        {
            "top_1_accuracy": float(
                metrics.top_k_accuracy_score(y_true=target.numpy(), y_score=prediction.numpy(), k=1)
            ),
            "top_1_error": 1.0
            - float(
                metrics.top_k_accuracy_score(y_true=target.numpy(), y_score=prediction.numpy(), k=1)
            ),
            "top_1_count": 100,
            "top_1_count_correct": int(
                metrics.top_k_accuracy_score(
                    y_true=target.numpy(), y_score=prediction.numpy(), k=1, normalize=False
                )
            ),
            "top_1_count_incorrect": 100
            - int(
                metrics.top_k_accuracy_score(
                    y_true=target.numpy(), y_score=prediction.numpy(), k=1, normalize=False
                )
            ),
            "top_3_accuracy": float(
                metrics.top_k_accuracy_score(y_true=target.numpy(), y_score=prediction.numpy(), k=3)
            ),
            "top_3_error": 1.0
            - float(
                metrics.top_k_accuracy_score(y_true=target.numpy(), y_score=prediction.numpy(), k=3)
            ),
            "top_3_count": 100,
            "top_3_count_correct": int(
                metrics.top_k_accuracy_score(
                    y_true=target.numpy(), y_score=prediction.numpy(), k=3, normalize=False
                )
            ),
            "top_3_count_incorrect": 100
            - int(
                metrics.top_k_accuracy_score(
                    y_true=target.numpy(), y_score=prediction.numpy(), k=3, normalize=False
                )
            ),
        },
    )
