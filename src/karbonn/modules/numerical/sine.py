r"""Contain modules to encode numerical values using cosine and sine
functions."""

from __future__ import annotations

__all__ = ["CosSinNumericalEncoder", "prepare_tensor_param"]

from typing import TYPE_CHECKING

import torch
from torch.nn import Module, Parameter

if TYPE_CHECKING:
    from torch import Tensor


class CosSinNumericalEncoder(Module):
    r"""Implement a frequency/phase-shift numerical encoder where the
    periodic functions are cosine and sine.

    Args:
        frequency: The initial frequency values. This input should be
            a tensor of shape ``(n_features, feature_size // 2)`` or
            ``(feature_size // 2,)``.
        phase_shift: The initial phase-shift values. This input should
            be a tensor of shape ``(n_features, feature_size // 2)`` or
            ``(feature_size // 2,)``.
        learnable: If ``True`` the frequencies and phase-shift
            parameters are learnable, otherwise they are frozen.

    Shape:
        - Input: ``(*, n_features)``, where ``*`` means any number of
            dimensions.
        - Output: ``(*, n_features, feature_size)``,  where ``*`` has
            the same shape as the input.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.modules import CosSinNumericalEncoder
    >>> # Example with 1 feature
    >>> m = CosSinNumericalEncoder(
    ...     frequency=torch.tensor([[1.0, 2.0, 4.0]]),
    ...     phase_shift=torch.zeros(1, 3),
    ... )
    >>> m
    CosSinNumericalEncoder(frequency=(1, 6), phase_shift=(1, 6), learnable=False)
    >>> out = m(torch.tensor([[0.0], [1.0], [2.0], [3.0]]))
    >>> out
    tensor([[[ 0.0000,  0.0000,  0.0000,  1.0000,  1.0000,  1.0000]],
            [[ 0.8415,  0.9093, -0.7568,  0.5403, -0.4161, -0.6536]],
            [[ 0.9093, -0.7568,  0.9894, -0.4161, -0.6536, -0.1455]],
            [[ 0.1411, -0.2794, -0.5366, -0.9900,  0.9602,  0.8439]]])
    >>> # Example with 2 features
    >>> m = CosSinNumericalEncoder(
    ...     frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 4.0, 6.0]]),
    ...     phase_shift=torch.zeros(2, 3),
    ... )
    >>> m
    CosSinNumericalEncoder(frequency=(2, 6), phase_shift=(2, 6), learnable=False)
    >>> out = m(torch.tensor([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]]))
    >>> out
    tensor([[[ 0.0000,  0.0000,  0.0000,  1.0000,  1.0000,  1.0000],
             [-0.2794, -0.5366, -0.7510,  0.9602,  0.8439,  0.6603]],
            [[ 0.8415,  0.9093, -0.7568,  0.5403, -0.4161, -0.6536],
             [-0.7568,  0.9894, -0.5366, -0.6536, -0.1455,  0.8439]],
            [[ 0.9093, -0.7568,  0.9894, -0.4161, -0.6536, -0.1455],
             [ 0.9093, -0.7568, -0.2794, -0.4161, -0.6536,  0.9602]],
            [[ 0.1411, -0.2794, -0.5366, -0.9900,  0.9602,  0.8439],
             [ 0.0000,  0.0000,  0.0000,  1.0000,  1.0000,  1.0000]]])

    ```
    """

    def __init__(self, frequency: Tensor, phase_shift: Tensor, learnable: bool = False) -> None:
        super().__init__()
        frequency = prepare_tensor_param(frequency, name="frequency")
        self.frequency = Parameter(frequency.repeat(1, 2), requires_grad=learnable)

        phase_shift = prepare_tensor_param(phase_shift, name="phase_shift")
        self.phase_shift = Parameter(phase_shift.repeat(1, 2), requires_grad=learnable)
        if self.frequency.shape != self.phase_shift.shape:
            msg = (
                f"'frequency' and 'phase_shift' shapes do not match: {self.frequency.shape} "
                f"vs {self.phase_shift.shape}"
            )
            raise RuntimeError(msg)

        self._half_size = int(frequency.shape[1])

    @property
    def input_size(self) -> int:
        r"""Return the input feature size."""
        return self.frequency.shape[0]

    @property
    def output_size(self) -> int:
        r"""Return the output feature size."""
        return self.frequency.shape[1]

    def extra_repr(self) -> str:
        return (
            f"frequency={tuple(self.frequency.shape)}, "
            f"phase_shift={tuple(self.phase_shift.shape)}, "
            f"learnable={self.frequency.requires_grad}"
        )

    def forward(self, scalar: Tensor) -> Tensor:
        features = scalar.unsqueeze(dim=-1).mul(self.frequency).add(self.phase_shift)
        return torch.cat(
            (features[..., : self._half_size].sin(), features[..., self._half_size :].cos()),
            dim=-1,
        )


def prepare_tensor_param(tensor: Tensor, name: str) -> Tensor:
    r"""Prepare a tensor parameter to be a 2d tensor.

    Args:
        tensor: The tensor to prepare.
        name: The name associated to the tensor.

    Returns:
        The prepared tensor.

    Raises:
        RuntimeError: if the input tensor is not a 1d or 2d tensor.
    """
    if tensor.ndim == 1:
        tensor = tensor.view(1, -1)
    if tensor.ndim != 2:
        msg = f"Incorrect shape for '{name}': {tensor.shape}"
        raise RuntimeError(msg)
    return tensor
