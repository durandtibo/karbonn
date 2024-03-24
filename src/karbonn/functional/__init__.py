r"""Contain functional implementation of some modules."""

from __future__ import annotations

__all__ = [
    "safe_exp",
    "safe_log",
    "reduce_loss",
    "check_loss_reduction_strategy",
]

from karbonn.functional.activations import safe_exp, safe_log
from karbonn.functional.reduction import check_loss_reduction_strategy, reduce_loss
