from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices

from karbonn import ReLUn, SquaredReLU

SIZES = ((1, 1), (2, 3), (2, 3, 4), (2, 3, 4, 5))

###########################
#     Tests for ReLUn     #
###########################


def test_relun_str() -> None:
    assert str(ReLUn()).startswith("ReLUn(")


@pytest.mark.parametrize("device", get_available_devices())
def test_relun_forward(device: str) -> None:
    device = torch.device(device)
    module = ReLUn().to(device=device)
    assert module(torch.arange(-1, 4, dtype=torch.float, device=device)).equal(
        torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float, device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_relun_forward_max_value_2(device: str) -> None:
    device = torch.device(device)
    module = ReLUn(max=2).to(device=device)
    assert module(torch.arange(-1, 4, dtype=torch.float, device=device)).equal(
        torch.tensor([0.0, 0.0, 1.0, 2.0, 2.0], dtype=torch.float, device=device)
    )


@pytest.mark.parametrize("size", SIZES)
def test_relun_forward_size(size: tuple[int, ...]) -> None:
    module = ReLUn()
    out = module(torch.randn(*size))
    assert out.shape == size
    assert out.dtype == torch.float


#################################
#     Tests for SquaredReLU     #
#################################


@pytest.mark.parametrize("device", get_available_devices())
def test_squared_relu_forward(device: str) -> None:
    device = torch.device(device)
    module = SquaredReLU().to(device=device)
    assert module(torch.arange(-1, 4, dtype=torch.float, device=device)).equal(
        torch.tensor([0.0, 0.0, 1.0, 4.0, 9.0], dtype=torch.float, device=device)
    )


@pytest.mark.parametrize("size", SIZES)
def test_squared_relu_forward_size(size: tuple[int, ...]) -> None:
    module = SquaredReLU()
    out = module(torch.randn(*size))
    assert out.shape == size
    assert out.dtype == torch.float
