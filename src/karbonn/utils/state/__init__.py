r"""Contain state implementations for metrics."""

from __future__ import annotations

__all__ = ["AverageState", "EmptyStateError"]

from karbonn.utils.state.average import AverageState
from karbonn.utils.state.exception import EmptyStateError
