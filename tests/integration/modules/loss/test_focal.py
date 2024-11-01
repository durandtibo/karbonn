from __future__ import annotations

import torch
from torch import nn

from karbonn.modules import BinaryFocalLossWithLogits
from karbonn.utils import is_loss_decreasing_with_adam

##############################################
#     Tests of BinaryFocalLossWithLogits     #
##############################################


def test_binary_focal_loss_with_logits() -> None:
    assert is_loss_decreasing_with_adam(
        module=nn.Sequential(nn.Linear(4, 4), nn.Sigmoid()),
        criterion=BinaryFocalLossWithLogits(),
        feature=torch.rand(8, 4),
        target=torch.randint(0, 2, size=(8, 4), dtype=torch.float),
    )
