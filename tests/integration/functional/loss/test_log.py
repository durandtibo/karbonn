from __future__ import annotations

import torch
from torch import nn

from karbonn.functional import log_cosh_loss, msle_loss
from karbonn.utils import is_loss_decreasing_with_sgd

###################################
#     Tests for log_cosh_loss     #
###################################


def test_log_cosh_loss_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(4, 2),
        criterion=log_cosh_loss,
        feature=torch.rand(4, 4),
        target=torch.rand(4, 2),
    )


###############################
#     Tests for msle_loss     #
###############################


def test_msle_loss_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Sequential(nn.Linear(4, 2), nn.Sigmoid()),
        criterion=msle_loss,
        feature=torch.rand(4, 4),
        target=torch.rand(4, 2),
    )
