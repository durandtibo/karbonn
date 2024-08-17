from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices
from minrecord import MaxScalarRecord, MinScalarRecord

from karbonn.metric import (
    BinaryConfusionMatrix,
    CategoricalConfusionMatrix,
    EmptyMetricError,
)
from karbonn.testing import sklearn_available
from karbonn.utils.imports import is_sklearn_available

if is_sklearn_available():
    from sklearn import metrics

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
            "count": 4,
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
            "count": 4,
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
            "count": 4,
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
            "count": 6,
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
            "count": 6,
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
            "count": 6,
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
            "count": 2,
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_binary_confusion_matrix_value_track_count_false(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = BinaryConfusionMatrix(track_count=False).to(device=device)
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
        },
    )


def test_binary_confusion_matrix_value_empty() -> None:
    with pytest.raises(EmptyMetricError, match="BinaryConfusionMatrix is empty"):
        BinaryConfusionMatrix().value()


def test_binary_confusion_matrix_reset() -> None:
    metric = BinaryConfusionMatrix()
    metric(torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 0, 1]))
    assert metric._tracker.count == 4
    metric.reset()
    assert metric._tracker.count == 0


def test_binary_confusion_get_records() -> None:
    metric = BinaryConfusionMatrix()
    assert objects_are_equal(
        metric.get_records(),
        (
            MaxScalarRecord(name="accuracy"),
            MaxScalarRecord(name="balanced_accuracy"),
            MaxScalarRecord(name="jaccard_index"),
            MaxScalarRecord(name="precision"),
            MaxScalarRecord(name="recall"),
            MaxScalarRecord(name="true_negative_rate"),
            MaxScalarRecord(name="true_negative"),
            MaxScalarRecord(name="true_positive_rate"),
            MaxScalarRecord(name="true_positive"),
            MinScalarRecord(name="false_negative_rate"),
            MinScalarRecord(name="false_negative"),
            MinScalarRecord(name="false_positive_rate"),
            MinScalarRecord(name="false_positive"),
            MaxScalarRecord(name="f1_score"),
        ),
    )


def test_binary_confusion_get_records_betas() -> None:
    metric = BinaryConfusionMatrix(betas=(0.5, 1, 2))
    assert objects_are_equal(
        metric.get_records(),
        (
            MaxScalarRecord(name="accuracy"),
            MaxScalarRecord(name="balanced_accuracy"),
            MaxScalarRecord(name="jaccard_index"),
            MaxScalarRecord(name="precision"),
            MaxScalarRecord(name="recall"),
            MaxScalarRecord(name="true_negative_rate"),
            MaxScalarRecord(name="true_negative"),
            MaxScalarRecord(name="true_positive_rate"),
            MaxScalarRecord(name="true_positive"),
            MinScalarRecord(name="false_negative_rate"),
            MinScalarRecord(name="false_negative"),
            MinScalarRecord(name="false_positive_rate"),
            MinScalarRecord(name="false_positive"),
            MaxScalarRecord(name="f0.5_score"),
            MaxScalarRecord(name="f1_score"),
            MaxScalarRecord(name="f2_score"),
        ),
    )


@pytest.mark.parametrize("prefix", ["prefix_", "suffix/"])
@pytest.mark.parametrize("suffix", ["_prefix", "/suffix"])
def test_binary_confusion_get_records_prefix_suffix(prefix: str, suffix: str) -> None:
    metric = BinaryConfusionMatrix()
    assert objects_are_equal(
        metric.get_records(prefix, suffix),
        (
            MaxScalarRecord(name=f"{prefix}accuracy{suffix}"),
            MaxScalarRecord(name=f"{prefix}balanced_accuracy{suffix}"),
            MaxScalarRecord(name=f"{prefix}jaccard_index{suffix}"),
            MaxScalarRecord(name=f"{prefix}precision{suffix}"),
            MaxScalarRecord(name=f"{prefix}recall{suffix}"),
            MaxScalarRecord(name=f"{prefix}true_negative_rate{suffix}"),
            MaxScalarRecord(name=f"{prefix}true_negative{suffix}"),
            MaxScalarRecord(name=f"{prefix}true_positive_rate{suffix}"),
            MaxScalarRecord(name=f"{prefix}true_positive{suffix}"),
            MinScalarRecord(name=f"{prefix}false_negative_rate{suffix}"),
            MinScalarRecord(name=f"{prefix}false_negative{suffix}"),
            MinScalarRecord(name=f"{prefix}false_positive_rate{suffix}"),
            MinScalarRecord(name=f"{prefix}false_positive{suffix}"),
            MaxScalarRecord(name=f"{prefix}f1_score{suffix}"),
        ),
    )


@sklearn_available
def test_binary_confusion_matrix_value_sklearn() -> None:
    metric = BinaryConfusionMatrix()
    prediction = torch.randint(0, 2, size=(100,))
    target = torch.randint(0, 2, size=(100,))
    metric(prediction=prediction, target=target)
    assert objects_are_allclose(
        metric.value(),
        {
            "accuracy": float(
                metrics.accuracy_score(y_pred=prediction.numpy(), y_true=target.numpy())
            ),
            "balanced_accuracy": float(
                metrics.balanced_accuracy_score(y_pred=prediction.numpy(), y_true=target.numpy())
            ),
            "false_negative": torch.logical_and(prediction == 0, target == 1).sum().item(),
            "false_negative_rate": torch.logical_and(prediction == 0, target == 1).sum().item()
            / (target == 1).sum().item(),
            "false_positive": torch.logical_and(prediction == 1, target == 0).sum().item(),
            "false_positive_rate": torch.logical_and(prediction == 1, target == 0).sum().item()
            / (target == 0).sum().item(),
            "jaccard_index": float(
                metrics.jaccard_score(y_pred=prediction.numpy(), y_true=target.numpy())
            ),
            "precision": float(
                metrics.precision_score(y_pred=prediction.numpy(), y_true=target.numpy())
            ),
            "recall": float(metrics.recall_score(y_pred=prediction.numpy(), y_true=target.numpy())),
            "true_negative": torch.logical_and(prediction == target, target == 0).sum().item(),
            "true_negative_rate": torch.logical_and(prediction == target, target == 0).sum().item()
            / (target == 0).sum().item(),
            "true_positive": torch.logical_and(prediction == target, target == 1).sum().item(),
            "true_positive_rate": torch.logical_and(prediction == target, target == 1).sum().item()
            / (target == 1).sum().item(),
            "f1_score": float(
                metrics.fbeta_score(beta=1, y_pred=prediction.numpy(), y_true=target.numpy())
            ),
            "count": 100,
        },
    )


################################################
#     Tests for CategoricalConfusionMatrix     #
################################################


def test_categorical_confusion_matrix_str() -> None:
    assert str(CategoricalConfusionMatrix(num_classes=3)).startswith("CategoricalConfusionMatrix(")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_confusion_matrix_forward_correct(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = CategoricalConfusionMatrix(num_classes=3).to(device=device)
    metric.train(mode)
    metric(
        torch.tensor([0, 1, 0, 1, 2], device=device),
        torch.tensor([0, 1, 0, 1, 2], device=device),
    )
    assert objects_are_equal(
        metric.value(),
        {
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1_score": 1.0,
            "micro_precision": 1.0,
            "micro_recall": 1.0,
            "micro_f1_score": 1.0,
            "weighted_precision": 1.0,
            "weighted_recall": 1.0,
            "weighted_f1_score": 1.0,
            "count": 5,
            "precision": torch.tensor([1.0, 1.0, 1.0]),
            "recall": torch.tensor([1.0, 1.0, 1.0]),
            "f1_score": torch.tensor([1.0, 1.0, 1.0]),
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_confusion_matrix_forward_incorrect(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = CategoricalConfusionMatrix(num_classes=3).to(device=device)
    metric.train(mode)
    metric(
        torch.tensor([0, 1, 0, 1, 2], device=device),
        torch.tensor([1, 0, 1, 2, 0], device=device),
    )
    assert objects_are_equal(
        metric.value(),
        {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1_score": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1_score": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1_score": 0.0,
            "count": 5,
            "precision": torch.tensor([0.0, 0.0, 0.0]),
            "recall": torch.tensor([0.0, 0.0, 0.0]),
            "f1_score": torch.tensor([0.0, 0.0, 0.0]),
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_confusion_matrix_forward_betas(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = CategoricalConfusionMatrix(num_classes=3, betas=(0.5, 1, 2)).to(device=device)
    metric.train(mode)
    metric(
        torch.tensor([0, 1, 0, 1, 2], device=device),
        torch.tensor([0, 1, 0, 1, 2], device=device),
    )
    assert objects_are_equal(
        metric.value(),
        {
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f0.5_score": 1.0,
            "macro_f1_score": 1.0,
            "macro_f2_score": 1.0,
            "micro_precision": 1.0,
            "micro_recall": 1.0,
            "micro_f0.5_score": 1.0,
            "micro_f1_score": 1.0,
            "micro_f2_score": 1.0,
            "weighted_precision": 1.0,
            "weighted_recall": 1.0,
            "weighted_f0.5_score": 1.0,
            "weighted_f1_score": 1.0,
            "weighted_f2_score": 1.0,
            "count": 5,
            "precision": torch.tensor([1.0, 1.0, 1.0]),
            "recall": torch.tensor([1.0, 1.0, 1.0]),
            "f0.5_score": torch.tensor([1.0, 1.0, 1.0]),
            "f1_score": torch.tensor([1.0, 1.0, 1.0]),
            "f2_score": torch.tensor([1.0, 1.0, 1.0]),
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_confusion_matrix_forward_2d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = CategoricalConfusionMatrix(num_classes=3).to(device=device)
    metric.train(mode)
    metric(
        torch.tensor([[0, 1], [0, 2], [1, 0]], device=device),
        torch.tensor([[0, 1], [0, 2], [1, 0]], device=device),
    )
    assert objects_are_equal(
        metric.value(),
        {
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1_score": 1.0,
            "micro_precision": 1.0,
            "micro_recall": 1.0,
            "micro_f1_score": 1.0,
            "weighted_precision": 1.0,
            "weighted_recall": 1.0,
            "weighted_f1_score": 1.0,
            "count": 6,
            "precision": torch.tensor([1.0, 1.0, 1.0]),
            "recall": torch.tensor([1.0, 1.0, 1.0]),
            "f1_score": torch.tensor([1.0, 1.0, 1.0]),
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype_prediction", DTYPES)
@pytest.mark.parametrize("dtype_target", DTYPES)
def test_categorical_confusion_matrix_forward_dtypes(
    device: str,
    mode: bool,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = CategoricalConfusionMatrix(num_classes=3).to(device=device)
    metric.train(mode)
    metric(
        torch.tensor([[0, 1], [0, 2], [1, 0]], device=device, dtype=dtype_prediction),
        torch.tensor([[0, 1], [0, 2], [1, 0]], device=device, dtype=dtype_target),
    )
    assert objects_are_equal(
        metric.value(),
        {
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1_score": 1.0,
            "micro_precision": 1.0,
            "micro_recall": 1.0,
            "micro_f1_score": 1.0,
            "weighted_precision": 1.0,
            "weighted_recall": 1.0,
            "weighted_f1_score": 1.0,
            "count": 6,
            "precision": torch.tensor([1.0, 1.0, 1.0]),
            "recall": torch.tensor([1.0, 1.0, 1.0]),
            "f1_score": torch.tensor([1.0, 1.0, 1.0]),
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_confusion_matrix_forward_multiple_batches(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = CategoricalConfusionMatrix(num_classes=3).to(device=device)
    metric.train(mode)
    metric(
        torch.tensor([0, 1, 0, 1], device=device),
        torch.tensor([0, 1, 0, 1], device=device),
    )
    metric(torch.tensor([2, 0], device=device), torch.tensor([2, 0], device=device))
    assert objects_are_equal(
        metric.value(),
        {
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1_score": 1.0,
            "micro_precision": 1.0,
            "micro_recall": 1.0,
            "micro_f1_score": 1.0,
            "weighted_precision": 1.0,
            "weighted_recall": 1.0,
            "weighted_f1_score": 1.0,
            "count": 6,
            "precision": torch.tensor([1.0, 1.0, 1.0]),
            "recall": torch.tensor([1.0, 1.0, 1.0]),
            "f1_score": torch.tensor([1.0, 1.0, 1.0]),
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_confusion_matrix_forward_multiple_batches_with_reset(
    device: str, mode: bool
) -> None:
    device = torch.device(device)
    metric = CategoricalConfusionMatrix(num_classes=3).to(device=device)
    metric.train(mode)
    metric(
        torch.tensor([0, 1, 0, 1], device=device),
        torch.tensor([0, 1, 0, 1], device=device),
    )
    metric.reset()
    metric(
        torch.tensor([1, 0, 2], device=device),
        torch.tensor([1, 0, 2], device=device),
    )
    assert objects_are_equal(
        metric.value(),
        {
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1_score": 1.0,
            "micro_precision": 1.0,
            "micro_recall": 1.0,
            "micro_f1_score": 1.0,
            "weighted_precision": 1.0,
            "weighted_recall": 1.0,
            "weighted_f1_score": 1.0,
            "count": 3,
            "precision": torch.tensor([1.0, 1.0, 1.0]),
            "recall": torch.tensor([1.0, 1.0, 1.0]),
            "f1_score": torch.tensor([1.0, 1.0, 1.0]),
        },
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_confusion_matrix_value_track_count(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = CategoricalConfusionMatrix(num_classes=3, track_count=False).to(device=device)
    metric.train(mode)
    metric(
        torch.tensor([0, 1, 0, 1, 2], device=device),
        torch.tensor([0, 1, 0, 1, 2], device=device),
    )
    assert objects_are_equal(
        metric.value(),
        {
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1_score": 1.0,
            "micro_precision": 1.0,
            "micro_recall": 1.0,
            "micro_f1_score": 1.0,
            "weighted_precision": 1.0,
            "weighted_recall": 1.0,
            "weighted_f1_score": 1.0,
            "precision": torch.tensor([1.0, 1.0, 1.0]),
            "recall": torch.tensor([1.0, 1.0, 1.0]),
            "f1_score": torch.tensor([1.0, 1.0, 1.0]),
        },
    )


def test_categorical_confusion_matrix_value_empty() -> None:
    with pytest.raises(EmptyMetricError, match="CategoricalConfusionMatrix is empty"):
        CategoricalConfusionMatrix(num_classes=3).value()


def test_categorical_confusion_matrix_reset() -> None:
    metric = CategoricalConfusionMatrix(num_classes=3)
    metric(torch.tensor([0, 1, 0, 1, 2]), torch.tensor([0, 1, 0, 1, 2]))
    assert metric._tracker.count == 5
    metric.reset()
    assert metric._tracker.count == 0


def test_categorical_confusion_get_records() -> None:
    metric = CategoricalConfusionMatrix(num_classes=3)
    assert objects_are_equal(
        metric.get_records(),
        (
            MaxScalarRecord(name="accuracy"),
            MaxScalarRecord(name="balanced_accuracy"),
            MaxScalarRecord(name="macro_precision"),
            MaxScalarRecord(name="macro_recall"),
            MaxScalarRecord(name="micro_precision"),
            MaxScalarRecord(name="micro_recall"),
            MaxScalarRecord(name="weighted_precision"),
            MaxScalarRecord(name="weighted_recall"),
            MaxScalarRecord(name="macro_f1_score"),
            MaxScalarRecord(name="micro_f1_score"),
            MaxScalarRecord(name="weighted_f1_score"),
        ),
    )


def test_categorical_confusion_get_records_betas() -> None:
    metric = CategoricalConfusionMatrix(num_classes=3, betas=(0.5, 1, 2))
    assert objects_are_equal(
        metric.get_records(),
        (
            MaxScalarRecord(name="accuracy"),
            MaxScalarRecord(name="balanced_accuracy"),
            MaxScalarRecord(name="macro_precision"),
            MaxScalarRecord(name="macro_recall"),
            MaxScalarRecord(name="micro_precision"),
            MaxScalarRecord(name="micro_recall"),
            MaxScalarRecord(name="weighted_precision"),
            MaxScalarRecord(name="weighted_recall"),
            MaxScalarRecord(name="macro_f0.5_score"),
            MaxScalarRecord(name="micro_f0.5_score"),
            MaxScalarRecord(name="weighted_f0.5_score"),
            MaxScalarRecord(name="macro_f1_score"),
            MaxScalarRecord(name="micro_f1_score"),
            MaxScalarRecord(name="weighted_f1_score"),
            MaxScalarRecord(name="macro_f2_score"),
            MaxScalarRecord(name="micro_f2_score"),
            MaxScalarRecord(name="weighted_f2_score"),
        ),
    )


@pytest.mark.parametrize("prefix", ["prefix_", "suffix/"])
@pytest.mark.parametrize("suffix", ["_prefix", "/suffix"])
def test_categorical_confusion_get_records_prefix_suffix(prefix: str, suffix: str) -> None:
    metric = CategoricalConfusionMatrix(num_classes=3)
    assert objects_are_equal(
        metric.get_records(prefix, suffix),
        (
            MaxScalarRecord(name=f"{prefix}accuracy{suffix}"),
            MaxScalarRecord(name=f"{prefix}balanced_accuracy{suffix}"),
            MaxScalarRecord(name=f"{prefix}macro_precision{suffix}"),
            MaxScalarRecord(name=f"{prefix}macro_recall{suffix}"),
            MaxScalarRecord(name=f"{prefix}micro_precision{suffix}"),
            MaxScalarRecord(name=f"{prefix}micro_recall{suffix}"),
            MaxScalarRecord(name=f"{prefix}weighted_precision{suffix}"),
            MaxScalarRecord(name=f"{prefix}weighted_recall{suffix}"),
            MaxScalarRecord(name=f"{prefix}macro_f1_score{suffix}"),
            MaxScalarRecord(name=f"{prefix}micro_f1_score{suffix}"),
            MaxScalarRecord(name=f"{prefix}weighted_f1_score{suffix}"),
        ),
    )


@sklearn_available
def test_categorical_confusion_matrix_value_sklearn() -> None:
    metric = CategoricalConfusionMatrix(num_classes=5)
    prediction = torch.randint(0, 5, size=(100,))
    target = torch.randint(0, 5, size=(100,))
    metric(prediction=prediction, target=target)
    assert objects_are_allclose(
        metric.value(),
        {
            "accuracy": float(
                metrics.accuracy_score(y_pred=prediction.numpy(), y_true=target.numpy())
            ),
            "balanced_accuracy": float(
                metrics.balanced_accuracy_score(y_pred=prediction.numpy(), y_true=target.numpy())
            ),
            "macro_precision": float(
                metrics.precision_score(
                    y_pred=prediction.numpy(), y_true=target.numpy(), average="macro"
                )
            ),
            "macro_recall": float(
                metrics.recall_score(
                    y_pred=prediction.numpy(), y_true=target.numpy(), average="macro"
                )
            ),
            "macro_f1_score": float(
                metrics.f1_score(y_pred=prediction.numpy(), y_true=target.numpy(), average="macro")
            ),
            "micro_precision": float(
                metrics.precision_score(
                    y_pred=prediction.numpy(), y_true=target.numpy(), average="micro"
                )
            ),
            "micro_recall": float(
                metrics.recall_score(
                    y_pred=prediction.numpy(), y_true=target.numpy(), average="micro"
                )
            ),
            "micro_f1_score": float(
                metrics.f1_score(y_pred=prediction.numpy(), y_true=target.numpy(), average="micro")
            ),
            "weighted_precision": float(
                metrics.precision_score(
                    y_pred=prediction.numpy(), y_true=target.numpy(), average="weighted"
                )
            ),
            "weighted_recall": float(
                metrics.recall_score(
                    y_pred=prediction.numpy(), y_true=target.numpy(), average="weighted"
                )
            ),
            "weighted_f1_score": float(
                metrics.f1_score(
                    y_pred=prediction.numpy(), y_true=target.numpy(), average="weighted"
                )
            ),
            "count": 100,
            "precision": torch.from_numpy(
                metrics.precision_score(
                    y_pred=prediction.numpy(), y_true=target.numpy(), average=None
                )
            ).float(),
            "recall": torch.from_numpy(
                metrics.recall_score(y_pred=prediction.numpy(), y_true=target.numpy(), average=None)
            ).float(),
            "f1_score": torch.from_numpy(
                metrics.f1_score(y_pred=prediction.numpy(), y_true=target.numpy(), average=None)
            ).float(),
        },
    )
