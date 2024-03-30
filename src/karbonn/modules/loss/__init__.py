r"""Contain loss functions."""

from __future__ import annotations

__all__ = ["BinaryFocalLoss", "GeneralRobustRegressionLoss", "binary_focal_loss"]

from karbonn.modules.loss.focal import BinaryFocalLoss, binary_focal_loss
from karbonn.modules.loss.general_robust import GeneralRobustRegressionLoss
