from __future__ import annotations

import torch
from torch import nn

from karbonn.modules import GeneralRobustRegressionLoss
from karbonn.utils import is_loss_decreasing_with_adam

SIZES = (1, 2)
TOLERANCE = 1e-6


################################################
#     Tests of GeneralRobustRegressionLoss     #
################################################


def test_general_robust_regression_loss() -> None:
    assert is_loss_decreasing_with_adam(
        module=nn.Sequential(nn.Linear(4, 4), nn.Sigmoid()),
        criterion=GeneralRobustRegressionLoss(),
        feature=torch.rand(8, 4),
        target=torch.randint(0, 2, size=(8, 4), dtype=torch.float),
    )
