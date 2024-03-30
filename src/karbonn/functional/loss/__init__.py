r"""Contain functional implementation of some loss functions."""

from __future__ import annotations

__all__ = ["asinh_mse_loss", "general_robust_regression_loss"]

from karbonn.functional.loss.asinh import asinh_mse_loss
from karbonn.functional.loss.general_robust import general_robust_regression_loss
