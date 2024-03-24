from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal
from coola.utils.tensor import get_available_devices

from karbonn.functional import check_loss_reduction_strategy, reduce_loss

DTYPES = (torch.long, torch.float)
SHAPES = [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]

###################################################
#     Tests for check_loss_reduction_strategy     #
###################################################


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_check_loss_reduction_strategy_valid(reduction: str) -> None:
    check_loss_reduction_strategy(reduction)


def test_check_loss_reduction_strategy_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect reduction: incorrect."):
        check_loss_reduction_strategy("incorrect")


#################################
#     Tests for reduce_loss     #
#################################


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
def test_reduce_loss_mean(device: str, dtype: torch.dtype) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        reduce_loss(
            torch.tensor([[3, 2, 1], [1, 0, -1]], dtype=dtype, device=device), reduction="mean"
        ),
        torch.tensor(1.0, dtype=torch.float, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_reduce_loss_mean_shape(device: str, dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        reduce_loss(torch.ones(*shape, dtype=dtype, device=device), reduction="mean"),
        torch.tensor(1.0, dtype=torch.float, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
def test_reduce_loss_sum(device: str, dtype: torch.dtype) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        reduce_loss(
            torch.tensor([[3, 2, 1], [1, 0, -1]], dtype=dtype, device=device), reduction="sum"
        ),
        torch.tensor(6.0, dtype=dtype, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_reduce_loss_sum_shape(device: str, dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        reduce_loss(torch.zeros(*shape, dtype=dtype, device=device), reduction="sum"),
        torch.tensor(0.0, dtype=dtype, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
def test_reduce_loss_none(device: str, dtype: torch.dtype) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        reduce_loss(
            torch.tensor([[3, 2, 1], [1, 0, -1]], dtype=dtype, device=device), reduction="none"
        ),
        torch.tensor([[3, 2, 1], [1, 0, -1]], dtype=dtype, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_reduce_loss_none_shape(device: str, dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        reduce_loss(torch.ones(*shape, dtype=dtype, device=device), reduction="none"),
        torch.ones(*shape, dtype=dtype, device=device),
    )


def test_reduce_loss_reduction_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect reduction: incorrect."):
        reduce_loss(torch.ones(2, 2), reduction="incorrect")
