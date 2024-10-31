r"""Contain functional implementation of some modules."""

from __future__ import annotations

__all__ = [
    "absolute_error",
    "absolute_relative_error",
    "asinh_mse_loss",
    "asinh_smooth_l1_loss",
    "check_loss_reduction_strategy",
    "general_robust_regression_loss",
    "log_cosh_loss",
    "msle_loss",
    "poisson_regression_loss",
    "quantile_regression_loss",
    "reduce_loss",
    "relative_loss",
    "safe_exp",
    "safe_log",
    "sigmoid_focal_loss",
    "sigmoid_poly1_loss",
    "symmetric_absolute_relative_error",
    "binary_poly1_loss",
]

from karbonn.functional.activations import safe_exp, safe_log
from karbonn.functional.error import (
    absolute_error,
    absolute_relative_error,
    symmetric_absolute_relative_error,
)
from karbonn.functional.loss import (
    asinh_mse_loss,
    asinh_smooth_l1_loss,
    binary_poly1_loss,
    general_robust_regression_loss,
    log_cosh_loss,
    msle_loss,
    poisson_regression_loss,
    quantile_regression_loss,
    relative_loss,
    sigmoid_focal_loss,
    sigmoid_poly1_loss,
)
from karbonn.functional.reduction import check_loss_reduction_strategy, reduce_loss
