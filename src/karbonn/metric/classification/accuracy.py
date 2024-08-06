r"""Contain accuracy metrics."""

from __future__ import annotations

__all__ = ["BinaryAccuracy"]

import logging
from typing import TYPE_CHECKING

from torch.nn import Identity, Module

from karbonn.metric.state import AccuracyState, BaseState
from karbonn.metric.state_ import BaseStateMetric
from karbonn.utils import setup_module

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


class BinaryAccuracy(BaseStateMetric):
    r"""Implement the binary accuracy metric.

    Args:
        state: The metric state or its configuration. If ``None``,
            ``AccuracyState`` is instantiated.
        transform: The transformation applied on the predictions to
            generate the predicted binary labels. If ``None``, the
            identity module is used. The transform module must take
            a single input tensor and output a single input tensor
            with the same shape as the input.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.metric import BinaryAccuracy
    >>> metric = BinaryAccuracy()
    >>> metric
    BinaryAccuracy(
      (state): AccuracyState(
          (tracker): MeanTensorTracker(count=0, total=0.0)
          (track_num_predictions): True
        )
      (transform): Identity()
    )
    >>> metric(torch.zeros(4), torch.ones(4))
    >>> metric.value()
    {'accuracy': 0.0, 'num_predictions': 4}
    >>> metric(torch.ones(4), torch.ones(4))
    >>> metric.value()
    {'accuracy': 0.5, 'num_predictions': 8}
    >>> metric.reset()
    >>> metric(torch.ones(4), torch.ones(4))
    >>> metric.value("bin_acc_")
    {'bin_acc_accuracy': 1.0, 'bin_acc_num_predictions': 4}

    ```
    """

    def __init__(
        self,
        state: BaseState | dict | None = None,
        transform: Module | dict | None = None,
    ) -> None:
        super().__init__(state=state or AccuracyState())
        self.transform = setup_module(transform or Identity())

    def forward(self, prediction: Tensor, target: Tensor) -> None:
        r"""Update the binary accuracy metric given a mini-batch of
        examples.

        Args:
            prediction: The predictions or the predicted labels.
                This input must be a ``torch.Tensor`` of  shape
                ``(d0, d1, ..., dn)`` or ``(d0, d1, ..., dn, 1)``
                and type bool or long or float. If the input is the
                predictions/scores, then the ``transform`` module
                should be set to transform the predictions/scores
                to binary labels where the values are ``0`` or ``1``.
            target: The binary targets where the values are  ``0`` or
                ``1``. This input must be a ``torch.Tensor``  of shape
                ``(d0, d1, ..., dn)`` or  ``(d0, d1, ..., dn, 1)``
                and type bool or long or float.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.metric import BinaryAccuracy
        >>> metric = BinaryAccuracy()
        >>> metric(torch.zeros(4), torch.ones(4))
        >>> metric.value()
        {'accuracy': 0.0, 'num_predictions': 4}

        ```
        """
        prediction = self.transform(prediction)
        self._state.update(prediction.eq(target.view_as(prediction)))
