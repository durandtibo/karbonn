from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices

from karbonn import Squeeze

#############################
#     Tests for Squeeze     #
#############################


def test_squeeze_str() -> None:
    assert str(Squeeze()).startswith("Squeeze(")


@pytest.mark.parametrize("device", get_available_devices())
def test_squeeze_forward_dim_none(device: str) -> None:
    device = torch.device(device)
    module = Squeeze().to(device=device)
    assert module(torch.ones(2, 1, 3, 1, 4, device=device)).equal(
        torch.ones(2, 3, 4, device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_squeeze_forward_dim_1(device: str) -> None:
    device = torch.device(device)
    module = Squeeze(dim=1).to(device=device)
    assert module(torch.ones(2, 1, 3, 1, 4, device=device)).equal(
        torch.ones(2, 3, 1, 4, device=device),
    )
