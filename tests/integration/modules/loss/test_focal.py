from __future__ import annotations

import pytest
import torch
from torch import nn

from karbonn.modules import binary_focal_loss
from karbonn.utils import is_loss_decreasing_with_adam

SIZES = (1, 2)
TOLERANCE = 1e-6


####################################
#     Tests of BinaryFocalLoss     #
####################################


@pytest.mark.parametrize("logits", {True, False})
def test_binary_focal_loss(logits: bool) -> None:
    assert is_loss_decreasing_with_adam(
        module=nn.Sequential(nn.Linear(4, 4), nn.Sigmoid()),
        criterion=binary_focal_loss(logits=logits),
        feature=torch.rand(8, 4),
        target=torch.randint(0, 2, size=(8, 4), dtype=torch.float),
    )
