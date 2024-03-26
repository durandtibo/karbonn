from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices
from torch.nn import Identity, Linear, ReLU, Sequential

from karbonn import ResidualBlock

SIZES = (1, 2)


###################################
#     Tests for ResidualBlock     #
###################################


def test_residual_block_residual() -> None:
    assert isinstance(ResidualBlock(residual=Linear(4, 4)).residual, Linear)


def test_residual_block_skip_default() -> None:
    assert isinstance(ResidualBlock(residual=Linear(4, 4)).skip, Identity)


def test_residual_block_skip() -> None:
    assert isinstance(ResidualBlock(residual=Linear(4, 4), skip=Linear(4, 4)).skip, Linear)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_residual_block_forward(device: str, batch_size: int, mode: bool) -> None:
    device = torch.device(device)
    module = ResidualBlock(residual=Linear(4, 4)).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, 4, device=device))
    assert out.shape == (batch_size, 4)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_residual_block_forward_skip(device: str, batch_size: int, mode: bool) -> None:
    device = torch.device(device)
    module = ResidualBlock(
        residual=Sequential(Linear(4, 8), ReLU(), Linear(8, 4)), skip=Linear(4, 4)
    ).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, 4, device=device))
    assert out.shape == (batch_size, 4)
    assert out.device == device
    assert out.dtype == torch.float
