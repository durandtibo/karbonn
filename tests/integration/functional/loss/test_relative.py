from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss

from karbonn.functional import relative_loss
from karbonn.functional.loss import (
    arithmetical_mean_indicator,
    classical_relative_indicator,
    geometric_mean_indicator,
    reversed_relative_indicator,
)
from karbonn.utils import is_loss_decreasing_with_sgd

if TYPE_CHECKING:
    from collections.abc import Callable


LOSSES = [
    partial(mse_loss, reduction="none"),
    partial(smooth_l1_loss, reduction="none"),
    partial(l1_loss, reduction="none"),
]
INDICATORS = [
    arithmetical_mean_indicator,
    classical_relative_indicator,
    geometric_mean_indicator,
    reversed_relative_indicator,
]
REDUCTIONS = ["mean", "sum"]


###################################
#     Tests for relative_loss     #
###################################


@pytest.mark.parametrize("base_loss", LOSSES)
@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_relative_loss_loss_decreasing(
    base_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    indicator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    reduction: str,
) -> None:
    def my_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return relative_loss(
            loss=base_loss(prediction, target),
            indicator=indicator(prediction, target),
            reduction=reduction,
        )

    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(8, 8),
        criterion=my_loss,
        feature=torch.randn(16, 8),
        target=torch.randn(16, 8),
        num_iterations=10,
    )
