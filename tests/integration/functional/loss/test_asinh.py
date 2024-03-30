from __future__ import annotations

import torch
from torch import nn

from karbonn.functional import asinh_mse_loss
from karbonn.utils import is_loss_decreasing_with_sgd

####################################
#     Tests for asinh_mse_loss     #
####################################


def test_asinh_mse_loss_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(4, 2),
        criterion=asinh_mse_loss,
        feature=torch.rand(4, 4),
        target=torch.rand(4, 2),
    )
