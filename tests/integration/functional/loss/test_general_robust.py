from __future__ import annotations

import torch
from torch import nn

from karbonn.functional import general_robust_regression_loss
from karbonn.utils import is_loss_decreasing_with_sgd

####################################################
#     Tests for general_robust_regression_loss     #
####################################################


def test_general_robust_regression_loss_decreasing() -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(4, 2),
        criterion=general_robust_regression_loss,
        feature=torch.rand(4, 4),
        target=torch.rand(4, 2),
    )
