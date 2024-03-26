r"""Contain functional implementation of some modules."""

from __future__ import annotations

__all__ = [
    "absolute_error",
    "absolute_relative_error",
    "check_loss_reduction_strategy",
    "reduce_loss",
    "safe_exp",
    "safe_log",
    "symmetric_absolute_relative_error",
]

from karbonn.functional.activations import safe_exp, safe_log
from karbonn.functional.error import (
    absolute_error,
    absolute_relative_error,
    symmetric_absolute_relative_error,
)
from karbonn.functional.reduction import check_loss_reduction_strategy, reduce_loss
