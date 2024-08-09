r"""Contain confusion matrix metrics for binary and categorical
labels."""

from __future__ import annotations

__all__ = ["BinaryConfusionMatrix"]

import logging
from typing import TYPE_CHECKING

from coola.utils import repr_mapping

from karbonn.metric.state_ import BaseStateMetric
from karbonn.utils.tracker import BinaryConfusionMatrix as BinaryConfusionMatrixState

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from karbonn.metric.state import BaseState

logger = logging.getLogger(__name__)


class BinaryConfusionMatrix(BaseStateMetric):
    r"""Implement a confusion matrix metric for binary labels.

    Args:
        betas (sequence, optional): Specifies the betas used to
            compute the f-beta score. Default: ``(1,)``

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.metric import BinaryConfusionMatrix
    >>> metric = BinaryConfusionMatrix()
    >>> metric
    BinaryConfusionMatrix(
      (betas): (1,)
      (state): BinaryConfusionMatrix(num_classes=2, num_predictions=0)
    )
    >>> metric(torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 0, 1]))
    >>> metric.value()
    {'accuracy': 1.0,
     'balanced_accuracy': 1.0,
     'false_negative_rate': 0.0,
     'false_negative': 0,
     'false_positive_rate': 0.0,
     'false_positive': 0,
     'jaccard_index': 1.0,
     'num_predictions': 4,
     'precision': 1.0,
     'recall': 1.0,
     'true_negative_rate': 1.0,
     'true_negative': 2,
     'true_positive_rate': 1.0,
     'true_positive': 2,
     'f1_score': 1.0}
    >>> metric(torch.tensor([1, 0]), torch.tensor([1, 0]))
    >>> metric.value()
    {'accuracy': 1.0,
     'balanced_accuracy': 1.0,
     'false_negative_rate': 0.0,
     'false_negative': 0,
     'false_positive_rate': 0.0,
     'false_positive': 0,
     'jaccard_index': 1.0,
     'num_predictions': 6,
     'precision': 1.0,
     'recall': 1.0,
     'true_negative_rate': 1.0,
     'true_negative': 3,
     'true_positive_rate': 1.0,
     'true_positive': 3,
     'f1_score': 1.0}
    >>> metric.reset()
    >>> metric(torch.tensor([1, 0]), torch.tensor([1, 0]))
    >>> metric.value()
    {'accuracy': 1.0,
     'balanced_accuracy': 1.0,
     'false_negative_rate': 0.0,
     'false_negative': 0,
     'false_positive_rate': 0.0,
     'false_positive': 0,
     'jaccard_index': 1.0,
     'num_predictions': 2,
     'precision': 1.0,
     'recall': 1.0,
     'true_negative_rate': 1.0,
     'true_negative': 1,
     'true_positive_rate': 1.0,
     'true_positive': 1,
     'f1_score': 1.0}

    ```
    """

    def __init__(
        self,
        betas: Sequence[int | float] = (1,),
        state: BaseState | dict | None = None,
    ) -> None:
        super().__init__(state=state or BinaryConfusionMatrixState())
        self._betas = tuple(betas)

    def extra_repr(self) -> str:
        return repr_mapping(
            {
                "betas": self._betas,
                "state": str(self._state),
            }
        )

    def forward(self, prediction: Tensor, target: Tensor) -> None:
        r"""Update the confusion matrix metric given a mini-batch of
        examples.

        Args:
            prediction: The predicted labels where the values are ``0`` or
                ``1``. This input must be a ``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` or ``(d0, d1, ..., dn, 1)``
                and type long or float.
            target: The binary targets where the values are ``0`` or
                ``1``. This input must be a ``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` or  ``(d0, d1, ..., dn, 1)``
                and type bool or long or float.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.metric import BinaryConfusionMatrix
        >>> metric = BinaryConfusionMatrix()
        >>> metric(torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 0, 1]))
        >>> metric.value()
        {'accuracy': 1.0,
         'balanced_accuracy': 1.0,
         'false_negative_rate': 0.0,
         'false_negative': 0,
         'false_positive_rate': 0.0,
         'false_positive': 0,
         'jaccard_index': 1.0,
         'num_predictions': 4,
         'precision': 1.0,
         'recall': 1.0,
         'true_negative_rate': 1.0,
         'true_negative': 2,
         'true_positive_rate': 1.0,
         'true_positive': 2,
         'f1_score': 1.0}

        ```
        """
        self._state.update(prediction.flatten(), target.flatten())


    def value(self, prefix: str = "", suffix: str = "") -> dict:
        return self._state.value(prefix=prefix, suffix=suffix)