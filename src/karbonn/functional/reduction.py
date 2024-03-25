r"""Contain loss reduction functions."""

from __future__ import annotations

__all__ = ["check_loss_reduction_strategy", "reduce_loss"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

VALID_REDUCTIONS = ("none", "mean", "sum")


def check_loss_reduction_strategy(reduction: str) -> None:
    r"""Check if the provided reduction ia a valid loss reduction.

    The valid reduction values are ``'mean'``, ``'none'``,  and
    ``'sum'``.

    Args:
        reduction: The reduction strategy to check.

    Raises:
        ValueError: if the provided reduction is not valid.

    Example usage:

    ```pycon

    >>> from karbonn.functional import check_loss_reduction_strategy
    >>> check_loss_reduction_strategy("mean")

    ```
    """
    if reduction not in VALID_REDUCTIONS:
        msg = f"Incorrect reduction: {reduction}. The valid reductions are: {VALID_REDUCTIONS}"
        raise ValueError(msg)


def reduce_loss(tensor: Tensor, reduction: str) -> Tensor:
    r"""Return the reduced loss.

    This function is designed to be used with loss functions.

    Args:
        tensor: The input tensor to reduce.
        reduction: The reduction strategy. The valid values are
            ``'mean'``, ``'none'``,  and ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum will be divided by the number of elements in the
            input, ``'sum'``: the output will be summed.

    Returns:
        The reduced tensor. The shape of the tensor depends on the
            reduction strategy.

    Raises:
        ValueError: if the reduction is not valid.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.functional import reduce_loss
    >>> tensor = torch.arange(6).view(2, 3)
    >>> reduce_loss(tensor, "none")
    tensor([[0, 1, 2],
            [3, 4, 5]])
    >>> reduce_loss(tensor, "sum")
    tensor(15)
    >>> reduce_loss(tensor, "mean")
    tensor(2.5000)

    ```
    """
    if reduction == "mean":
        return tensor.float().mean()
    if reduction == "sum":
        return tensor.sum()
    if reduction == "none":
        return tensor
    msg = f"Incorrect reduction: {reduction}. The valid reductions are {VALID_REDUCTIONS}."
    raise ValueError(msg)
