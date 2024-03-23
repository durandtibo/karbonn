r"""Contain utility functions."""

from __future__ import annotations

__all__ = [
    "is_loss_decreasing",
    "is_loss_decreasing_with_adam",
    "is_loss_decreasing_with_sgd",
    "is_module_config",
    "module_mode",
    "setup_module",
    "top_module_mode",
]
from karbonn.utils.factory import is_module_config, setup_module
from karbonn.utils.loss import (
    is_loss_decreasing,
    is_loss_decreasing_with_adam,
    is_loss_decreasing_with_sgd,
)
from karbonn.utils.mode import module_mode, top_module_mode
