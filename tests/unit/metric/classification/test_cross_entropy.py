from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices
from minrecord import MinScalarRecord

from karbonn.metric import CategoricalCrossEntropy, EmptyMetricError
from karbonn.metric.state import BaseState, ExtendedErrorState, MeanErrorState

MODES = (True, False)
SIZES = (1, 2)
DTYPES = (torch.long, torch.float)

TOLERANCE = 1e-7


#############################################
#     Tests for CategoricalCrossEntropy     #
#############################################


def test_categorical_cross_entropy_str() -> None:
    assert str(CategoricalCrossEntropy()).startswith("CategoricalCrossEntropy(")


def test_categorical_cross_entropy_state_default() -> None:
    assert isinstance(CategoricalCrossEntropy().state, MeanErrorState)


def test_categorical_cross_entropy_state_extended() -> None:
    assert isinstance(
        CategoricalCrossEntropy(state=ExtendedErrorState()).state,
        ExtendedErrorState,
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", SIZES)
def test_categorical_cross_entropy_forward_correct(
    device: str, mode: bool, batch_size: int
) -> None:
    device = torch.device(device)
    metric = CategoricalCrossEntropy().to(device=device)
    metric.train(mode)
    prediction = torch.zeros(batch_size, 3, device=device)
    prediction[:, 0] = 1
    metric(prediction, torch.zeros(batch_size, device=device))
    assert objects_are_allclose(
        metric.value(),
        {"mean": 0.5514447139320511, "num_predictions": batch_size},
        atol=TOLERANCE,
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", SIZES)
def test_categorical_cross_entropy_forward_incorrect(
    device: str, mode: bool, batch_size: int
) -> None:
    device = torch.device(device)
    metric = CategoricalCrossEntropy().to(device=device)
    metric.train(mode)
    prediction = torch.zeros(batch_size, 3, device=device)
    prediction[:, 0] = 1
    metric(prediction, torch.ones(batch_size, device=device))
    assert objects_are_allclose(
        metric.value(),
        {"mean": 1.551444713932051, "num_predictions": batch_size},
        atol=TOLERANCE,
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_cross_entropy_forward_partially_correct(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = CategoricalCrossEntropy().to(device=device)
    metric.train(mode)
    metric(torch.eye(2, device=device), torch.zeros(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {"mean": 0.8132616875182228, "num_predictions": 2},
        atol=TOLERANCE,
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_cross_entropy_forward_prediction_2d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = CategoricalCrossEntropy().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 3, device=device), torch.zeros(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {"mean": 1.0986122886681098, "num_predictions": 2},
        atol=TOLERANCE,
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_cross_entropy_forward_prediction_2d_target_2d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = CategoricalCrossEntropy().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 3, device=device), torch.zeros(2, 1, device=device))
    assert objects_are_allclose(
        metric.value(),
        {"mean": 1.0986122886681098, "num_predictions": 2},
        atol=TOLERANCE,
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_cross_entropy_forward_prediction_3d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = CategoricalCrossEntropy().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 3, 3, device=device), torch.zeros(2, 3, device=device))
    assert objects_are_allclose(
        metric.value(),
        {"mean": 1.0986122886681098, "num_predictions": 6},
        atol=TOLERANCE,
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_cross_entropy_forward_prediction_4d(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = CategoricalCrossEntropy().to(device=device)
    metric.train(mode)
    metric(torch.ones(2, 3, 4, 3, device=device), torch.zeros(2, 3, 4, device=device))
    assert objects_are_allclose(
        metric.value(),
        {"mean": 1.0986122886681098, "num_predictions": 24},
        atol=TOLERANCE,
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype_target", DTYPES)
def test_categorical_cross_entropy_forward_dtypes(
    device: str,
    mode: bool,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = CategoricalCrossEntropy().to(device=device)
    metric.train(mode)
    metric(torch.eye(4, device=device), torch.arange(4, device=device, dtype=dtype_target))
    assert objects_are_allclose(
        metric.value(),
        {"mean": 0.7436683806286791, "num_predictions": 4},
        atol=TOLERANCE,
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_cross_entropy_forward_state(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = CategoricalCrossEntropy(state=ExtendedErrorState()).to(device=device)
    metric.train(mode)
    metric(torch.eye(4, device=device), torch.arange(4, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            "mean": 0.7436683806286791,
            "median": 0.7436683806286791,
            "min": 0.7436683806286791,
            "max": 0.7436683806286791,
            "sum": 2.9746735225147165,
            "std": 0.0,
            "num_predictions": 4,
        },
        atol=TOLERANCE,
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_cross_entropy_forward_multiple_batches(device: str, mode: bool) -> None:
    device = torch.device(device)
    metric = CategoricalCrossEntropy().to(device=device)
    metric.train(mode)
    metric(torch.eye(4, device=device), torch.arange(4, device=device))
    metric(torch.ones(2, 3, device=device), torch.ones(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {"mean": 0.8619830166418226, "num_predictions": 6},
        atol=TOLERANCE,
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", MODES)
def test_categorical_cross_entropy_forward_multiple_batches_with_reset(
    device: str, mode: bool
) -> None:
    device = torch.device(device)
    metric = CategoricalCrossEntropy().to(device=device)
    metric.train(mode)
    metric(torch.eye(4, device=device), torch.arange(4, device=device))
    metric.reset()
    metric(torch.ones(2, 3, device=device), torch.ones(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {"mean": 1.0986122886681098, "num_predictions": 2},
        atol=TOLERANCE,
    )


def test_categorical_cross_entropy_value_empty() -> None:
    with pytest.raises(EmptyMetricError, match="MeanErrorState is empty"):
        CategoricalCrossEntropy().value()


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("prefix", ["prefix_", "suffix/"])
@pytest.mark.parametrize("suffix", ["_prefix", "/suffix"])
def test_categorical_cross_entropy_value_prefix_suffix(
    device: str, prefix: str, suffix: str
) -> None:
    device = torch.device(device)
    metric = CategoricalCrossEntropy().to(device=device)
    metric(torch.eye(4, device=device), torch.arange(4, device=device))
    assert objects_are_allclose(
        metric.value(prefix, suffix),
        {f"{prefix}mean{suffix}": 0.7436683806286791, f"{prefix}num_predictions{suffix}": 4},
    )


def test_categorical_cross_entropy_reset() -> None:
    state = Mock(spec=BaseState)
    metric = CategoricalCrossEntropy(state)
    metric.reset()
    state.reset.assert_called_once_with()


def test_categorical_cross_entropy_get_records() -> None:
    metric = CategoricalCrossEntropy()
    assert objects_are_equal(
        metric.get_records(),
        (MinScalarRecord(name="mean"),),
    )


@pytest.mark.parametrize("prefix", ["prefix_", "suffix/"])
@pytest.mark.parametrize("suffix", ["_prefix", "/suffix"])
def test_categorical_cross_entropy_get_records_prefix_suffix(prefix: str, suffix: str) -> None:
    metric = CategoricalCrossEntropy()
    assert objects_are_equal(
        metric.get_records(prefix, suffix),
        (MinScalarRecord(name=f"{prefix}mean{suffix}"),),
    )
