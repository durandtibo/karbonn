r"""Contain the loss functions that use the inverse hyperbolic sine
(asinh) to transform the predictions or targets."""

from __future__ import annotations

__all__ = ["asinh_mse_loss"]

from typing import TYPE_CHECKING

from torch.nn.functional import mse_loss

if TYPE_CHECKING:
    import torch


def asinh_mse_loss(
    prediction: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    r"""Compute the mean squared error (MSE) on the inverse hyperbolic
    sine (asinh) transformed predictions and targets.

    It is a generalization of mean squared logarithmic error (MSLE)
    that works for real values. The ``asinh`` transformation is used
    instead of ``log1p`` because ``asinh`` works on negative values.

    Args:
        prediction: The predictions.
        target: The target values.
        reduction: The reduction strategy. The valid values are
            ``'mean'``, ``'none'``,  and ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum will be divided by the number of elements in the
            input, ``'sum'``: the output will be summed.

    Returns:
        The mean squared error (MSE) on the inverse hyperbolic sine
            (asinh) transformed predictions and targets. The shape of
            the tensor depends on the reduction strategy.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.functional import asinh_mse_loss
    >>> loss = asinh_mse_loss(torch.randn(2, 4, requires_grad=True), torch.randn(2, 4))
    >>> loss
    tensor(..., grad_fn=<MseLossBackward0>)
    >>> loss.backward()

    ```
    """
    return mse_loss(prediction.asinh(), target.asinh(), reduction=reduction)
