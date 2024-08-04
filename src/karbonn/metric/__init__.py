r"""Contain the metrics."""

from __future__ import annotations

__all__ = [
    "BaseMetric",
    "EmptyMetricError",
    "setup_metric",
    "BaseStateMetric",
    "AbsoluteError",
    "AbsoluteRelativeError",
    "SymmetricAbsoluteRelativeError",
]

from karbonn.metric.base import BaseMetric, EmptyMetricError, setup_metric
from karbonn.metric.regression import (
    AbsoluteError,
    AbsoluteRelativeError,
    SymmetricAbsoluteRelativeError,
)
from karbonn.metric.state_ import BaseStateMetric
