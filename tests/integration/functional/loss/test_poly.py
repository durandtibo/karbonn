from __future__ import annotations

import torch
from torch import nn

from karbonn.functional import binary_poly1_loss, binary_poly1_loss_with_logits
from karbonn.utils import is_loss_decreasing_with_adam


def test_binary_poly1_loss() -> None:
    assert is_loss_decreasing_with_adam(
        module=nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.Sigmoid()),
        criterion=binary_poly1_loss,
        feature=torch.randn(8, 4),
        target=torch.randint(0, 2, size=(8, 4), dtype=torch.float),
    )


def test_binary_poly1_loss_with_logits() -> None:
    assert is_loss_decreasing_with_adam(
        module=nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4)),
        criterion=binary_poly1_loss_with_logits,
        feature=torch.randn(8, 4),
        target=torch.randint(0, 2, size=(8, 4), dtype=torch.float),
    )
