r"""Contain the focal loss implementation."""

from __future__ import annotations

__all__ = ["BinaryFocalLoss", "binary_focal_loss"]

import torch
from torch import nn

from karbonn.functional import check_loss_reduction_strategy, reduce_loss
from karbonn.utils import setup_module


class BinaryFocalLoss(nn.Module):
    r"""Implementation of the binary Focal Loss.

    Based on "Focal Loss for Dense Object Detection"
    (https://arxiv.org/pdf/1708.02002.pdf)

    Args:
        loss: The binary cross entropy layer or another equivalent
            layer. To be used as in the original paper, this loss
            should not use reducton as the reduction is done in this
            class.
        alpha: The weighting factor, which must be in the range
            ``[0, 1]``.
        gamma: The focusing parameter, which must be positive
            (``>=0``).
        reduction: The reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be
            summed.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Target: ``(*)``, same shape as the input.
        - Output: scalar. If ``reduction`` is ``'none'``, then ``(*)``, same
          shape as input.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn import BinaryFocalLoss
    >>> criterion = BinaryFocalLoss(nn.BCEWithLogitsLoss(reduction="none"))
    >>> input = torch.randn(3, 2, requires_grad=True)
    >>> target = torch.rand(3, 2, requires_grad=False)
    >>> output = criterion(input, target)
    >>> output.backward()

    ```
    """

    def __init__(
        self,
        loss: nn.Module | dict,
        alpha: float = 0.5,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.loss = setup_module(loss)

        if 0 <= alpha <= 1:
            self._alpha = float(alpha)
        else:
            msg = f"Incorrect parameter alpha ({alpha}). The valid range of value is [0, 1]."
            raise ValueError(msg)

        if gamma >= 0:
            self._gamma = float(gamma)
        else:
            msg = f"Incorrect parameter gamma ({gamma}). Gamma has to be positive (>=0)."
            raise ValueError(msg)

        check_loss_reduction_strategy(reduction)
        self.reduction = reduction

    def extra_repr(self) -> str:
        return f"alpha={self._alpha}, gamma={self._gamma}"

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Compute the binary Focal Loss.

        Args:
            prediction: The predicted probabilities or the
                un-normalized scores.
            target: The targets where ``1`` (resp. ``0``) means a
                positive (resp. negative) example.

        Returns:
            ``torch.Tensor`` of type float: The loss value(s). The
                shape of the tensor depends on the reduction. If the
                reduction is ``mean`` or ``sum``, the tensor has a
                single scalar value. If the reduction is ``none``,
                the shape of the tensor is the same that the inputs.
        """
        loss = self.loss(prediction, target)
        pt = torch.exp(-loss)
        # alpha for positive samples, else 1-alpha
        alpha_t = self._alpha * target + (1 - self._alpha) * (1 - target)
        focal_loss = alpha_t * (1 - pt) ** self._gamma * loss
        return reduce_loss(focal_loss, self.reduction)


def binary_focal_loss(
    alpha: float = 0.5, gamma: float = 2.0, reduction: str = "mean", logits: bool = False
) -> BinaryFocalLoss:
    r"""Return an instantiated binary focal loss with a binary cross
    entropy loss.

    Args:
        alpha: The weighting factor, which must be in the range
            ``[0, 1]``.
        gamma: The focusing parameter, which must be positive
            (``>=0``).
        reduction: The reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be
            summed.
        logits: If ``True``, the ``torch.nn.BCEWithLogitsLoss`` is
            used, otherwise ``torch.nn.BCELoss`` is used.

    Returns:
        The instantiated binary focal loss.

    Example usage:

    ```pycon

    >>> from karbonn import binary_focal_loss
    >>> criterion = binary_focal_loss()
    >>> criterion
    BinaryFocalLoss(
      alpha=0.5, gamma=2.0
      (loss): BCELoss()
    )
    >>> criterion = binary_focal_loss(logits=True)
    >>> criterion
    BinaryFocalLoss(
      alpha=0.5, gamma=2.0
      (loss): BCEWithLogitsLoss()
    )

    ```
    """
    loss = nn.BCEWithLogitsLoss(reduction="none") if logits else nn.BCELoss(reduction="none")
    return BinaryFocalLoss(loss=loss, alpha=alpha, gamma=gamma, reduction=reduction)
