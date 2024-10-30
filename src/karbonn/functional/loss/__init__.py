r"""Contain functional implementation of some loss functions."""

from __future__ import annotations

__all__ = [
    "arithmetical_mean_indicator",
    "asinh_mse_loss",
    "asinh_smooth_l1_loss",
    "classical_relative_indicator",
    "general_robust_regression_loss",
    "geometric_mean_indicator",
    "log_cosh_loss",
    "maximum_mean_indicator",
    "minimum_mean_indicator",
    "moment_mean_indicator",
    "msle_loss",
    "poisson_regression_loss",
    "quantile_regression_loss",
    "relative_loss",
    "reversed_relative_indicator",
    "sigmoid_focal_loss",
    "sigmoid_bce_poly1_loss",
]

from karbonn.functional.loss.asinh import asinh_mse_loss, asinh_smooth_l1_loss
from karbonn.functional.loss.focal import sigmoid_focal_loss
from karbonn.functional.loss.general_robust import general_robust_regression_loss
from karbonn.functional.loss.log import log_cosh_loss, msle_loss
from karbonn.functional.loss.poisson import poisson_regression_loss
from karbonn.functional.loss.poly import sigmoid_bce_poly1_loss
from karbonn.functional.loss.quantile import quantile_regression_loss
from karbonn.functional.loss.relative import (
    arithmetical_mean_indicator,
    classical_relative_indicator,
    geometric_mean_indicator,
    maximum_mean_indicator,
    minimum_mean_indicator,
    moment_mean_indicator,
    relative_loss,
    reversed_relative_indicator,
)
