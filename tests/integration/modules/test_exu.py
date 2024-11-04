from __future__ import annotations

import torch
from torch import nn
from torch.nn import MSELoss

from karbonn.modules import ExU
from karbonn.utils import is_loss_decreasing_with_adam


def test_nlinear_loss() -> None:
    assert is_loss_decreasing_with_adam(
        module=nn.Sequential(
            ExU(in_features=4, out_features=8), nn.ReLU(), nn.Linear(in_features=8, out_features=6)
        ),
        criterion=MSELoss(),
        feature=torch.randn(8, 4),
        target=torch.randn(8, 6),
    )
