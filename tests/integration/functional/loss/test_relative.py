from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss

from karbonn.functional import relative_loss
from karbonn.functional.loss.relative import RelativeIndicatorRegistry
from karbonn.utils import is_loss_decreasing_with_sgd

if TYPE_CHECKING:
    from collections.abc import Callable

###################################
#     Tests for relative_loss     #
###################################


@pytest.mark.parametrize(
    "base_loss",
    [
        partial(mse_loss, reduction="none"),
        partial(smooth_l1_loss, reduction="none"),
        partial(l1_loss, reduction="none"),
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("indicator", RelativeIndicatorRegistry.available_indicators())
def test_relative_loss_loss_decreasing(
    base_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], reduction: str, indicator: str
) -> None:
    def my_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return relative_loss(
            loss=base_loss(prediction, target),
            prediction=prediction,
            target=target,
            reduction=reduction,
            indicator=indicator,
        )

    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(8, 8),
        criterion=my_loss,
        feature=torch.randn(16, 8),
        target=torch.randn(16, 8),
        num_iterations=10,
    )
