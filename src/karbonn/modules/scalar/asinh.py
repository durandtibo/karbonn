r"""Contain modules to encode scalar values using the inverse hyperbolic
sine (asinh)."""

from __future__ import annotations

__all__ = ["AsinhScalarEncoder"]

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from karbonn.utils.tensor import to_tensor

if TYPE_CHECKING:
    from collections.abc import Sequence


class AsinhScalarEncoder(Module):
    r"""Implement a scalar encoder using the inverse hyperbolic sine
    (asinh).

    Args:
        scale: The initial scale values.
        learnable: If ``True`` the scales are learnable,
            otherwise they are frozen.

    Shape:
        - Input: ``(*, 1)``, where ``*`` means any number of
            dimensions.
        - Output: ``(*, feature_size)``,  where ``*`` has the same
            shape as the input.

    Example usage:

    ```pycon
    >>> import torch
    >>> from karbonn import AsinhScalarEncoder
    >>> m = AsinhScalarEncoder(scale=torch.tensor([1.0, 2.0, 4.0]))
    >>> m
    AsinhScalarEncoder(dim=3, learnable=False)
    >>> out = m(torch.tensor([[0.0], [1.0], [2.0], [3.0]]))
    >>> out
    tensor([[0.0000, 0.0000, 0.0000],
            [0.8814, 1.4436, 2.0947],
            [1.4436, 2.0947, 2.7765],
            [1.8184, 2.4918, 3.1798]])

    ```
    """

    def __init__(self, scale: Tensor | Sequence[float], learnable: bool = False) -> None:
        super().__init__()
        self.scale = Parameter(to_tensor(scale), requires_grad=learnable)

    @property
    def input_size(self) -> int:
        r"""Return the input feature size."""
        return 1

    @property
    def output_size(self) -> int:
        r"""Return the output feature size."""
        return self.scale.shape[0]

    def extra_repr(self) -> str:
        return f"dim={self.scale.shape[0]}, learnable={self.scale.requires_grad}"

    def forward(self, scalar: Tensor) -> Tensor:
        return scalar.mul(self.scale).asinh()

    @classmethod
    def create_rand_scale(
        cls,
        dim: int,
        min_scale: float,
        max_scale: float,
        learnable: bool = False,
    ) -> AsinhScalarEncoder:
        r"""Create a ``AsinhScalarEncoder`` where the scales are
        uniformly initialized in the specified scale range.

        Args:
            dim: The dimension i.e. the number of scales.
            min_scale: The minimum scale.
            max_scale: The maximum scale.
            learnable: If ``True`` the scales are learnable,
                otherwise they are frozen.

        Returns:
            An instantiated ``AsinhScalarEncoder`` where the scales are
                uniformly initialized in a scale range.

        Example usage:

        ```pycon
        >>> import torch
        >>> from karbonn import AsinhScalarEncoder
        >>> m = AsinhScalarEncoder.create_rand_scale(dim=5, min_scale=1, max_scale=10)
        >>> m
        AsinhScalarEncoder(dim=5, learnable=False)

        ```
        """
        if dim < 1:
            msg = f"dim has to be greater or equal to 1 (received: {dim})"
            raise ValueError(msg)
        if min_scale <= 0:
            msg = f"min_scale has to be greater than 0 (received: {min_scale})"
            raise ValueError(msg)
        if max_scale < min_scale:
            msg = (
                f"max_scale has to be greater than min_scale {min_scale} "
                f"(received: {max_scale})"
            )
            raise ValueError(msg)
        return cls(
            scale=torch.rand(dim).mul(max_scale - min_scale).add(min_scale),
            learnable=learnable,
        )

    @classmethod
    def create_linspace_scale(
        cls,
        dim: int,
        min_scale: float,
        max_scale: float,
        learnable: bool = False,
    ) -> AsinhScalarEncoder:
        r"""Create a ``AsinhScalarEncoder`` where the scales are evenly
        spaced.

        Args:
            dim: The dimension i.e. the number of scales.
            min_scale: The minimum scale.
            max_scale: The maximum scale.
            learnable: If ``True`` the scales are learnable,
                otherwise they are frozen.

        Returns:
            An instantiated ``AsinhScalarEncoder`` where the scales are
                evenly spaced.

        Example usage:

        ```pycon
        >>> import torch
        >>> from karbonn import AsinhScalarEncoder
        >>> m = AsinhScalarEncoder.create_linspace_scale(dim=5, min_scale=1, max_scale=10)
        >>> m
        AsinhScalarEncoder(dim=5, learnable=False)

        ```
        """
        if dim < 1:
            msg = f"dim has to be greater or equal to 1 (received: {dim})"
            raise ValueError(msg)
        if min_scale <= 0:
            msg = f"min_scale has to be greater than 0 (received: {min_scale})"
            raise ValueError(msg)
        if max_scale < min_scale:
            msg = (
                f"max_scale has to be greater than min_scale {min_scale} "
                f"(received: {max_scale})"
            )
            raise ValueError(msg)
        return cls(
            scale=torch.linspace(start=min_scale, end=max_scale, steps=dim),
            learnable=learnable,
        )

    @classmethod
    def create_logspace_scale(
        cls,
        dim: int,
        min_scale: float,
        max_scale: float,
        learnable: bool = False,
    ) -> AsinhScalarEncoder:
        r"""Create a ``AsinhScalarEncoder`` where the scales are evenly
        spaced in the log space.

        Args:
            dim: The dimension i.e. the number of scales.
            min_scale: The minimum scale.
            max_scale: The maximum scale.
            learnable: If ``True`` the scales are learnable,
                otherwise they are frozen.

        Returns:
            An instantiated ``AsinhScalarEncoder`` where the scales are
                evenly spaced in the log space.

        Example usage:

        ```pycon
        >>> import torch
        >>> from karbonn import AsinhScalarEncoder
        >>> m = AsinhScalarEncoder.create_logspace_scale(dim=5, min_scale=1, max_scale=10)
        >>> m
        AsinhScalarEncoder(dim=5, learnable=False)

        ```
        """
        if dim < 1:
            msg = f"dim has to be greater or equal to 1 (received: {dim})"
            raise ValueError(msg)
        if min_scale <= 0:
            msg = f"min_scale has to be greater than 0 (received: {min_scale})"
            raise ValueError(msg)
        if max_scale < min_scale:
            msg = (
                f"max_scale has to be greater than min_scale {min_scale} "
                f"(received: {max_scale})"
            )
            raise ValueError(msg)
        return cls(
            scale=torch.logspace(start=math.log10(min_scale), end=math.log10(max_scale), steps=dim),
            learnable=learnable,
        )
