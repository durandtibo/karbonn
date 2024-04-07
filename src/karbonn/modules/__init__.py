r"""Contains some modules."""

from __future__ import annotations

__all__ = [
    "Asinh",
    "AsinhCosSinScalarEncoder",
    "AsinhMSELoss",
    "AsinhScalarEncoder",
    "AsinhSmoothL1Loss",
    "AverageFusion",
    "BinaryFocalLoss",
    "Clamp",
    "ConcatFusion",
    "CosSinScalarEncoder",
    "ExU",
    "Exp",
    "ExpSin",
    "Expm1",
    "Gaussian",
    "GeneralRobustRegressionLoss",
    "Laplacian",
    "Log",
    "Log1p",
    "MultiQuadratic",
    "MulticlassFlatten",
    "MultiplicationFusion",
    "Quadratic",
    "ReLUn",
    "RelativeLoss",
    "RelativeMSELoss",
    "RelativeSmoothL1Loss",
    "ResidualBlock",
    "SafeExp",
    "SafeLog",
    "Sin",
    "Sinh",
    "Snake",
    "SquaredReLU",
    "Squeeze",
    "SumFusion",
    "ToFloat",
    "ToLong",
    "TransformedLoss",
    "binary_focal_loss",
]

from karbonn.modules.activations import (
    Asinh,
    Exp,
    Expm1,
    ExpSin,
    Gaussian,
    Laplacian,
    Log,
    Log1p,
    MultiQuadratic,
    Quadratic,
    ReLUn,
    SafeExp,
    SafeLog,
    Sin,
    Sinh,
    Snake,
    SquaredReLU,
)
from karbonn.modules.clamp import Clamp
from karbonn.modules.dtype import ToFloat, ToLong
from karbonn.modules.exu import ExU
from karbonn.modules.fusion import (
    AverageFusion,
    ConcatFusion,
    MultiplicationFusion,
    SumFusion,
)
from karbonn.modules.loss import (
    AsinhMSELoss,
    AsinhSmoothL1Loss,
    BinaryFocalLoss,
    GeneralRobustRegressionLoss,
    RelativeLoss,
    RelativeMSELoss,
    RelativeSmoothL1Loss,
    TransformedLoss,
    binary_focal_loss,
)
from karbonn.modules.residual import ResidualBlock
from karbonn.modules.scalar import (
    AsinhCosSinScalarEncoder,
    AsinhScalarEncoder,
    CosSinScalarEncoder,
)
from karbonn.modules.shape import MulticlassFlatten, Squeeze
