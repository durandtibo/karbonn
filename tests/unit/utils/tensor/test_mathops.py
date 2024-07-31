from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices

from karbonn.utils.tensor import quantile

DTYPES = (torch.float, torch.long)


##############################
#     Tests for quantile     #
##############################


@pytest.mark.parametrize("dtype", DTYPES)
def test_quantile_input_dtype(dtype: torch.dtype) -> None:
    assert quantile(torch.arange(11).to(dtype=dtype), q=torch.tensor([0.1])).equal(
        torch.tensor([1], dtype=torch.float),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_quantile_output_dtype(dtype: torch.dtype) -> None:
    assert quantile(torch.arange(11), q=torch.tensor([0.1])).equal(
        torch.tensor([1], dtype=dtype),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_quantile_device(device: str) -> None:
    device = torch.device(device)
    assert quantile(torch.arange(11, device=device), q=torch.tensor([0.1])).equal(
        torch.tensor([1], device=device),
    )


def test_quantile_q_multiple() -> None:
    assert quantile(torch.arange(11), q=torch.tensor([0.1, 0.5, 0.9])).equal(
        torch.tensor([1, 5, 9], dtype=torch.float),
    )
