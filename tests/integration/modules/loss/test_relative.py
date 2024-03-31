from __future__ import annotations

import pytest
import torch
from torch import nn

from karbonn import RelativeLoss, RelativeMSELoss, RelativeSmoothL1Loss
from karbonn.functional.loss.relative import RelativeIndicatorRegistry
from karbonn.utils import is_loss_decreasing_with_sgd

##################################
#     Tests for RelativeLoss     #
##################################


@pytest.mark.parametrize(
    "criterion",
    [nn.MSELoss(reduction="none"), nn.SmoothL1Loss(reduction="none"), nn.L1Loss(reduction="none")],
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("indicator", RelativeIndicatorRegistry.available_indicators())
def test_relative_loss_loss_decreasing(
    criterion: nn.Module, reduction: str, indicator: str
) -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(8, 8),
        criterion=RelativeLoss(criterion=criterion, reduction=reduction, indicator=indicator),
        feature=torch.randn(16, 8),
        target=torch.randn(16, 8),
        num_iterations=10,
    )


#####################################
#     Tests for RelativeMSELoss     #
#####################################


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("indicator", RelativeIndicatorRegistry.available_indicators())
def test_relative_mse_loss_loss_decreasing(reduction: str, indicator: str) -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(8, 8),
        criterion=RelativeMSELoss(reduction=reduction, indicator=indicator),
        feature=torch.randn(16, 8),
        target=torch.randn(16, 8),
        num_iterations=10,
    )


##########################################
#     Tests for RelativeSmoothL1Loss     #
##########################################


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("indicator", RelativeIndicatorRegistry.available_indicators())
def test_relative_smooth_l1_loss_loss_decreasing(reduction: str, indicator: str) -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(8, 8),
        criterion=RelativeSmoothL1Loss(reduction=reduction, indicator=indicator),
        feature=torch.randn(16, 8),
        target=torch.randn(16, 8),
        num_iterations=10,
    )
