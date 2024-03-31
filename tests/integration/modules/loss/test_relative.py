from __future__ import annotations

import pytest
import torch
from torch import nn

from karbonn import RelativeLoss, RelativeMSELoss, RelativeSmoothL1Loss
from karbonn.modules.loss import (
    ArithmeticalMeanIndicator,
    BaseRelativeIndicator,
    ClassicalRelativeIndicator,
    GeometricMeanIndicator,
    MaximumMeanIndicator,
    MinimumMeanIndicator,
    ReversedRelativeIndicator,
)
from karbonn.utils import is_loss_decreasing_with_sgd

CRITERIA = [
    nn.MSELoss(reduction="none"),
    nn.SmoothL1Loss(reduction="none"),
    nn.L1Loss(reduction="none"),
]
INDICATORS = [
    ArithmeticalMeanIndicator(),
    ClassicalRelativeIndicator(),
    GeometricMeanIndicator(),
    MaximumMeanIndicator(),
    MinimumMeanIndicator(),
    ReversedRelativeIndicator(),
]
REDUCTIONS = ["mean", "sum"]


##################################
#     Tests for RelativeLoss     #
##################################


@pytest.mark.parametrize("criterion", CRITERIA)
@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_relative_loss_loss_decreasing(
    criterion: nn.Module, reduction: str, indicator: BaseRelativeIndicator
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


@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_relative_mse_loss_loss_decreasing(
    reduction: str, indicator: BaseRelativeIndicator
) -> None:
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


@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_relative_smooth_l1_loss_loss_decreasing(
    reduction: str, indicator: BaseRelativeIndicator
) -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(8, 8),
        criterion=RelativeSmoothL1Loss(reduction=reduction, indicator=indicator),
        feature=torch.randn(16, 8),
        target=torch.randn(16, 8),
        num_iterations=10,
    )
