from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal
from coola.utils.tensor import get_available_devices
from minrecord import MaxScalarRecord, MinScalarRecord

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


def test_top_k_accuracy_get_records() -> None:
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


def test_top_k_accuracy_get_records_betas() -> None:
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
def test_top_k_accuracy_get_records_prefix_suffix(prefix: str, suffix: str) -> None:
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
