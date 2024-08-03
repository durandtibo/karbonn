r"""Contain the accuracy-based metric states."""

from __future__ import annotations

__all__ = ["AccuracyState"]

from typing import TYPE_CHECKING

from coola.utils import str_indent, str_mapping
from minrecord import BaseRecord, MaxScalarRecord

from karbonn.metric.base import EmptyMetricError
from karbonn.metric.state.base import BaseState
from karbonn.utils.tracker import MeanTensorTracker

if TYPE_CHECKING:
    import torch


class AccuracyState(BaseState):
    r"""Implements a metric state to compute the accuracy.

    This state has a constant space complexity.

    Args:
        track_num_predictions: If ``True``, the state tracks and
            returns the number of predictions.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.metric.state import AccuracyState
    >>> state = AccuracyState()
    >>> state
    AccuracyState(
      (tracker): MeanTensorTracker(count=0, total=0.0)
    )
    >>> state.get_records()
    (MaxScalarRecord(name=accuracy, max_size=10, size=0),)
    >>> state.update(torch.eye(4))
    >>> state.value()
    {'accuracy': 0.25, 'num_predictions': 16}

    ```
    """

    def __init__(self, track_num_predictions: bool = True) -> None:
        self._tracker = MeanTensorTracker()
        self._track_num_predictions = bool(track_num_predictions)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"tracker": self._tracker}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    @property
    def num_predictions(self) -> int:
        return self._tracker.count

    def get_records(self, prefix: str = "", suffix: str = "") -> tuple[BaseRecord, ...]:
        return (MaxScalarRecord(name=f"{prefix}accuracy{suffix}"),)

    def reset(self) -> None:
        self._tracker.reset()

    def update(self, correct: torch.Tensor) -> None:
        r"""Update the metric state with a new tensor of errors.

        Args:
            correct: A tensor that indicates the correct predictions.
                ``1`` indicates a correct prediction and ``0``
                indicates a bad prediction.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.metric.state import AccuracyState
        >>> state = AccuracyState()
        >>> state.update(torch.eye(4))
        >>> state.value()
        {'accuracy': 0.25, 'num_predictions': 16}

        ```
        """
        self._tracker.update(correct.detach())

    def value(self, prefix: str = "", suffix: str = "") -> dict[str, int | float]:
        tracker = self._tracker.all_reduce()
        if not tracker.count:
            msg = f"{self.__class__.__qualname__} is empty"
            raise EmptyMetricError(msg)

        results = {f"{prefix}accuracy{suffix}": tracker.mean()}
        if self._track_num_predictions:
            results[f"{prefix}num_predictions{suffix}"] = tracker.count
        return results
