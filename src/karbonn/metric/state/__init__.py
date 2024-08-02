r"""Contain the metric states."""

from __future__ import annotations

__all__ = ["BaseState", "MeanErrorState"]

from karbonn.metric.state.base import BaseState
from karbonn.metric.state.error import MeanErrorState
