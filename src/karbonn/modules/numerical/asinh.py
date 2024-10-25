r"""Contain modules to encode numerical values using the inverse
hyperbolic sine (asinh)."""

from __future__ import annotations

__all__ = ["AsinhNumericalEncoder"]

from typing import TYPE_CHECKING

from torch.nn import Module, Parameter

if TYPE_CHECKING:
    from torch import Tensor


class AsinhNumericalEncoder(Module):
    r"""Implement a numerical encoder using the inverse hyperbolic sine
    (asinh).

    Args:
        scale: The initial scale values. This input should be a tensor
            of shape ``(n_features, feature_size)`` or
            ``(feature_size,)``.
        learnable: If ``True`` the scales are learnable,
            otherwise they are frozen.

    Shape:
        - Input: ``(*, n_features)``, where ``*`` means any number of
            dimensions.
        - Output: ``(*, n_features, feature_size)``,  where ``*`` has the same
            shape as the input.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.modules import AsinhNumericalEncoder
    >>> # Example with 1 feature
    >>> m = AsinhNumericalEncoder(scale=torch.tensor([[1.0, 2.0, 4.0]]))
    >>> m
    AsinhNumericalEncoder(shape=(1, 3), learnable=False)
    >>> out = m(torch.tensor([[0.0], [1.0], [2.0], [3.0]]))
    >>> out
    tensor([[[0.0000, 0.0000, 0.0000]],
            [[0.8814, 1.4436, 2.0947]],
            [[1.4436, 2.0947, 2.7765]],
            [[1.8184, 2.4918, 3.1798]]])
    >>> # Example with 2 features
    >>> m = AsinhNumericalEncoder(scale=torch.tensor([[1.0, 2.0, 4.0], [1.0, 3.0, 6.0]]))
    >>> m
    AsinhNumericalEncoder(shape=(2, 3), learnable=False)
    >>> out = m(torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))
    >>> out
    tensor([[[0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000]],
            [[0.8814, 1.4436, 2.0947], [0.8814, 1.8184, 2.4918]],
            [[1.4436, 2.0947, 2.7765], [1.4436, 2.4918, 3.1798]],
            [[1.8184, 2.4918, 3.1798], [1.8184, 2.8934, 3.5843]]])

    ```
    """

    def __init__(self, scale: Tensor, learnable: bool = False) -> None:
        super().__init__()
        if scale.ndim == 1:
            scale = scale.view(1, -1)
        if scale.ndim != 2:
            msg = f"Incorrect shape for scale: {scale.shape}"
            raise ValueError(msg)
        self.scale = Parameter(scale, requires_grad=learnable)

    @property
    def input_size(self) -> int:
        r"""Return the input feature size."""
        return self.scale.shape[0]

    @property
    def output_size(self) -> int:
        r"""Return the output feature size."""
        return self.scale.shape[1]

    def extra_repr(self) -> str:
        return f"shape={tuple(self.scale.shape)}, learnable={self.scale.requires_grad}"

    def forward(self, scalar: Tensor) -> Tensor:
        return scalar.unsqueeze(dim=-1).mul(self.scale).asinh()
