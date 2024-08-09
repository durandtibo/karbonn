from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal
from coola.utils.tensor import get_available_devices

from karbonn.metric import BinaryConfusionMatrix, EmptyMetricError

MODES = (True, False)
SIZES = (1, 2)
DTYPES = (torch.long, torch.float)


###########################################
#     Tests for BinaryConfusionMatrix     #
###########################################


def test_binary_confusion_matrix_str() -> None:
    assert str(BinaryConfusionMatrix()).startswith("BinaryConfusionMatrix(")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_binary_confusion_matrix_forward_correct(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = BinaryConfusionMatrix().to(device=device)
    metric.train(mode)
    metric(torch.tensor([0, 1, 0, 1], device=device), torch.tensor([0, 1, 0, 1], device=device))
    assert objects_are_equal(
        metric.value(),
        {
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "false_negative_rate": 0.0,
            "false_negative": 0,
            "false_positive_rate": 0.0,
            "false_positive": 0,
            "jaccard_index": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "true_negative_rate": 1.0,
            "true_negative": 2,
            "true_positive_rate": 1.0,
            "true_positive": 2,
            "f1_score": 1.0,
            "num_predictions": 4,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_binary_confusion_matrix_forward_incorrect(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = BinaryConfusionMatrix().to(device=device)
    metric.train(mode)
    metric(torch.tensor([0, 1, 0, 1], device=device), torch.tensor([1, 0, 1, 0], device=device))
    assert objects_are_equal(
        metric.value(),
        {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "false_negative": 2,
            "false_negative_rate": 1.0,
            "false_positive": 2,
            "false_positive_rate": 1.0,
            "jaccard_index": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "true_negative": 0,
            "true_negative_rate": 0.0,
            "true_positive": 0,
            "true_positive_rate": 0.0,
            "f1_score": 0.0,
            "num_predictions": 4,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_binary_confusion_matrix_forward_betas(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = BinaryConfusionMatrix(betas=(0.5, 1, 2)).to(device=device)
    metric.train(mode)
    metric(torch.tensor([0, 1, 0, 1], device=device), torch.tensor([0, 1, 0, 1], device=device))
    assert objects_are_equal(
        metric.value(),
        {
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "false_negative": 0,
            "false_negative_rate": 0.0,
            "false_positive": 0,
            "false_positive_rate": 0.0,
            "jaccard_index": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "true_negative": 2,
            "true_negative_rate": 1.0,
            "true_positive": 2,
            "true_positive_rate": 1.0,
            "f0.5_score": 1.0,
            "f1_score": 1.0,
            "f2_score": 1.0,
            "num_predictions": 4,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_binary_confusion_matrix_forward_2d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = BinaryConfusionMatrix().to(device=device)
    metric.train(mode)
    metric(
        torch.tensor([[0, 1], [0, 1], [1, 0]], device=device),
        torch.tensor([[0, 1], [0, 1], [1, 0]], device=device),
    )
    assert objects_are_equal(
        metric.value(),
        {
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "false_negative": 0,
            "false_negative_rate": 0.0,
            "false_positive": 0,
            "false_positive_rate": 0.0,
            "jaccard_index": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "true_negative": 3,
            "true_negative_rate": 1.0,
            "true_positive": 3,
            "true_positive_rate": 1.0,
            "f1_score": 1.0,
            "num_predictions": 6,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype_prediction", DTYPES)
@pytest.mark.parametrize("dtype_target", DTYPES)
def test_binary_confusion_matrix_forward_dtypes(
    device: str,
    mode: bool,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = BinaryConfusionMatrix().to(device=device)
    metric.train(mode)
    metric(
        torch.tensor([[0, 1], [0, 1], [1, 0]], device=device, dtype=dtype_prediction),
        torch.tensor([[0, 1], [0, 1], [1, 0]], device=device, dtype=dtype_target),
    )
    assert objects_are_equal(
        metric.value(),
        {
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "false_negative": 0,
            "false_negative_rate": 0.0,
            "false_positive": 0,
            "false_positive_rate": 0.0,
            "jaccard_index": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "true_negative": 3,
            "true_negative_rate": 1.0,
            "true_positive": 3,
            "true_positive_rate": 1.0,
            "f1_score": 1.0,
            "num_predictions": 6,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_binary_confusion_matrix_forward_multiple_batches(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = BinaryConfusionMatrix().to(device=device)
    metric.train(mode)
    metric(torch.tensor([0, 1, 0, 1], device=device), torch.tensor([0, 1, 0, 1], device=device))
    metric(torch.tensor([1, 0], device=device), torch.tensor([1, 0], device=device))
    assert objects_are_equal(
        metric.value(),
        {
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "false_negative": 0,
            "false_negative_rate": 0.0,
            "false_positive": 0,
            "false_positive_rate": 0.0,
            "jaccard_index": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "true_negative": 3,
            "true_negative_rate": 1.0,
            "true_positive": 3,
            "true_positive_rate": 1.0,
            "f1_score": 1.0,
            "num_predictions": 6,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_binary_confusion_matrix_forward_multiple_batches_with_reset(
    device: str, mode: bool
) -> None:
    device = torch.device(device)
    metric = BinaryConfusionMatrix().to(device=device)
    metric.train(mode)
    metric(torch.tensor([0, 1, 0, 1], device=device), torch.tensor([0, 1, 0, 1], device=device))
    metric.reset()
    metric(torch.tensor([1, 0], device=device), torch.tensor([1, 0], device=device))
    assert objects_are_equal(
        metric.value(),
        {
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "false_negative": 0,
            "false_negative_rate": 0.0,
            "false_positive": 0,
            "false_positive_rate": 0.0,
            "jaccard_index": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "true_negative": 1,
            "true_negative_rate": 1.0,
            "true_positive": 1,
            "true_positive_rate": 1.0,
            "f1_score": 1.0,
            "num_predictions": 2,
        },
    )


def test_binary_confusion_matrix_value_empty() -> None:
    with pytest.raises(EmptyMetricError, match="BinaryConfusionMatrix is empty"):
        BinaryConfusionMatrix().value()


def test_binary_confusion_matrix_reset() -> None:
    metric = BinaryConfusionMatrix()
    metric(torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 0, 1]))
    assert metric.state.num_predictions == 4
    metric.reset()
    assert metric.state.num_predictions == 0
