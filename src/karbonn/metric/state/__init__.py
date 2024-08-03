r"""Contain the metric states."""

from __future__ import annotations

__all__ = [
    "BaseState",
    "ErrorState",
    "MeanErrorState",
    "ExtendedErrorState",
    "is_state_config",
    "setup_state",
    "RootMeanErrorState",
]

from karbonn.metric.state.base import BaseState, is_state_config, setup_state
from karbonn.metric.state.error import (
    ErrorState,
    ExtendedErrorState,
    MeanErrorState,
    RootMeanErrorState,
)
