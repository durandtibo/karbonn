from __future__ import annotations

import pytest
import torch
from torch import nn

from karbonn import Asinh, Clamp, Exp, Expm1, Log, Log1p, SafeExp, SafeLog, Sin, Sinh
from karbonn.utils import is_loss_decreasing_with_adam


@pytest.mark.parametrize(
    "activation",
    [
        Asinh(),
        nn.Sequential(Clamp(max=5.0), Exp()),
        nn.Sequential(Clamp(max=5.0), Expm1()),
        nn.Sequential(Clamp(min=1.0, max=5.0), Log()),
        nn.Sequential(Clamp(min=0.0, max=5.0), Log1p()),
        SafeExp(),
        SafeLog(),
        Sin(),
        Sinh(),
    ],
)
def test_activation_is_loss_decreasing(activation: nn.Module) -> None:
    assert is_loss_decreasing_with_adam(
        module=nn.Sequential(nn.Linear(6, 6), activation),
        criterion=nn.MSELoss(),
        feature=torch.randn(16, 6),
        target=torch.randn(16, 6),
        num_iterations=2,
    )
