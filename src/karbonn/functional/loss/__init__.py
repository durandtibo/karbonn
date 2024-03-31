r"""Contain functional implementation of some loss functions."""

from __future__ import annotations

__all__ = [
    "arithmetical_mean_indicator",
    "asinh_mse_loss",
    "asinh_smooth_l1_loss",
    "classical_relative_indicator",
    "general_robust_regression_loss",
    "log_cosh_loss",
    "msle_loss",
    "relative_loss",
    "reversed_relative_indicator",
]

from karbonn.functional.loss.asinh import asinh_mse_loss, asinh_smooth_l1_loss
from karbonn.functional.loss.general_robust import general_robust_regression_loss
from karbonn.functional.loss.log import log_cosh_loss, msle_loss
from karbonn.functional.loss.relative import (
    arithmetical_mean_indicator,
    classical_relative_indicator,
    relative_loss,
    reversed_relative_indicator,
)
