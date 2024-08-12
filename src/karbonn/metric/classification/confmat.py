r"""Contain confusion tracker metrics for binary and categorical
labels."""

from __future__ import annotations

__all__ = ["BinaryConfusionMatrix"]

import logging
from typing import TYPE_CHECKING

from coola.utils import repr_mapping

from karbonn.metric.base import BaseMetric, EmptyMetricError
from karbonn.utils.tracker import BinaryConfusionMatrixTracker

if TYPE_CHECKING:
    from collections.abc import Sequence

    from minrecord import BaseRecord
    from torch import Tensor


logger = logging.getLogger(__name__)


class BinaryConfusionMatrix(BaseMetric):
    r"""Implement a confusion tracker metric for binary labels.

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
      (tracker): BinaryConfusionMatrixTracker(num_classes=2, count=0)
      (track_count): True
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
     'count': 4,
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
     'count': 6,
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
     'count': 2,
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
        tracker: BinaryConfusionMatrixTracker | None = None,
        track_count: bool = True,
    ) -> None:
        super().__init__()
        self._betas = tuple(betas)
        self._tracker = tracker or BinaryConfusionMatrixTracker()
        self._track_count = bool(track_count)

    def extra_repr(self) -> str:
        return repr_mapping(
            {
                "betas": self._betas,
                "tracker": str(self._tracker),
                "track_count": self._track_count,
            }
        )

    def forward(self, prediction: Tensor, target: Tensor) -> None:
        r"""Update the confusion tracker metric given a mini-batch of
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
         'count': 4,
         'precision': 1.0,
         'recall': 1.0,
         'true_negative_rate': 1.0,
         'true_negative': 2,
         'true_positive_rate': 1.0,
         'true_positive': 2,
         'f1_score': 1.0}

        ```
        """
        self._tracker.update(prediction.flatten(), target.flatten())

    def get_records(self, prefix: str = "", suffix: str = "") -> tuple[BaseRecord, ...]:
        return self._tracker.get_records(betas=self._betas, prefix=prefix, suffix=suffix)

    def reset(self) -> None:
        self._tracker.reset()

    def value(self, prefix: str = "", suffix: str = "") -> dict:
        tracker = self._tracker.all_reduce()
        if not tracker.count:
            msg = f"{self.__class__.__qualname__} is empty"
            raise EmptyMetricError(msg)

        results = tracker.compute_metrics(betas=self._betas, prefix=prefix, suffix=suffix)
        if not self._track_count:
            del results[f"{prefix}count{suffix}"]
        return results
