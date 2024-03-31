r"""Contain relative loss functions."""

from __future__ import annotations

__all__ = [
    "ArithmeticalMeanIndicator",
    "BaseRelativeIndicator",
    "ClassicalRelativeIndicator",
    "RelativeLoss",
    "RelativeMSELoss",
    "RelativeSmoothL1Loss",
    "ReversedRelativeIndicator",
]

from karbonn.modules.loss.relative.indicators import (
    ArithmeticalMeanIndicator,
    BaseRelativeIndicator,
    ClassicalRelativeIndicator,
    ReversedRelativeIndicator,
)
from karbonn.modules.loss.relative.relative import (
    RelativeLoss,
    RelativeMSELoss,
    RelativeSmoothL1Loss,
)
