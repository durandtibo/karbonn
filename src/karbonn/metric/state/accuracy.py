r"""Contain the accuracy-based metric states."""

from __future__ import annotations

__all__ = ["AccuracyState"]

from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping
from minrecord import BaseRecord, MaxScalarRecord, MinScalarRecord

from karbonn.metric.base import EmptyMetricError
from karbonn.metric.state.base import BaseState
from karbonn.utils.tracker import MeanTensorTracker

if TYPE_CHECKING:
    import torch


class AccuracyState(BaseState):
    r"""Implement a metric state to compute the accuracy.

    This state has a constant space complexity.

    Args:
        tracker: The initial mean value tracker.
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
      (track_num_predictions): True
    )
    >>> state.get_records()
    (MaxScalarRecord(name=accuracy, max_size=10, size=0),)
    >>> state.update(torch.eye(4))
    >>> state.value()
    {'accuracy': 0.25, 'num_predictions': 16}

    ```
    """

    def __init__(
        self, tracker: MeanTensorTracker | None = None, track_num_predictions: bool = True
    ) -> None:
        self._tracker = tracker or MeanTensorTracker()
        self._track_num_predictions = bool(track_num_predictions)

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping(
                {"tracker": self._tracker, "track_num_predictions": self._track_num_predictions}
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "tracker": self._tracker,
                    "num_predictions": f"{self.num_predictions:,}",
                    "track_num_predictions": self._track_num_predictions,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    @property
    def num_predictions(self) -> int:
        return self._tracker.count

    def clone(self) -> AccuracyState:
        return self.__class__(
            tracker=self._tracker, track_num_predictions=self._track_num_predictions
        )

    def equal(self, other: Any) -> bool:
        if not isinstance(other, AccuracyState):
            return False
        return self._track_num_predictions == other._track_num_predictions and self._tracker.equal(
            other._tracker
        )

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


class ExtendedAccuracyState(BaseState):
    r"""Implement a metric state to compute the accuracy and other
    metrics.

    This state has a constant space complexity.

    Args:
        tracker: The initial mean value tracker.
        track_num_predictions: If ``True``, the state tracks and
            returns the number of predictions.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.metric.state import ExtendedAccuracyState
    >>> state = ExtendedAccuracyState()
    >>> state
    ExtendedAccuracyState(
      (tracker): MeanTensorTracker(count=0, total=0.0)
      (track_num_predictions): True
    )
    >>> state.get_records()
    (MaxScalarRecord(name=accuracy, max_size=10, size=0),
     MinScalarRecord(name=error, max_size=10, size=0),
     MaxScalarRecord(name=num_correct_predictions, max_size=10, size=0),
     MinScalarRecord(name=num_incorrect_predictions, max_size=10, size=0))
    >>> state.update(torch.eye(4))
    >>> state.value()
    {'accuracy': 0.25,
     'error': 0.75,
     'num_correct_predictions': 4,
     'num_incorrect_predictions': 12,
     'num_predictions': 16}

    ```
    """

    def __init__(
        self, tracker: MeanTensorTracker | None = None, track_num_predictions: bool = True
    ) -> None:
        self._tracker = tracker or MeanTensorTracker()
        self._track_num_predictions = bool(track_num_predictions)

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping(
                {"tracker": self._tracker, "track_num_predictions": self._track_num_predictions}
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "tracker": self._tracker,
                    "num_predictions": f"{self.num_predictions:,}",
                    "track_num_predictions": self._track_num_predictions,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    @property
    def num_predictions(self) -> int:
        return self._tracker.count

    def clone(self) -> ExtendedAccuracyState:
        return self.__class__(
            tracker=self._tracker, track_num_predictions=self._track_num_predictions
        )

    def equal(self, other: Any) -> bool:
        if not isinstance(other, ExtendedAccuracyState):
            return False
        return self._track_num_predictions == other._track_num_predictions and self._tracker.equal(
            other._tracker
        )

    def get_records(self, prefix: str = "", suffix: str = "") -> tuple[BaseRecord, ...]:
        return (
            MaxScalarRecord(name=f"{prefix}accuracy{suffix}"),
            MinScalarRecord(name=f"{prefix}error{suffix}"),
            MaxScalarRecord(name=f"{prefix}num_correct_predictions{suffix}"),
            MinScalarRecord(name=f"{prefix}num_incorrect_predictions{suffix}"),
        )

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
        >>> from karbonn.metric.state import ExtendedAccuracyState
        >>> state = ExtendedAccuracyState()
        >>> state.update(torch.eye(4))
        >>> state.value()
        {'accuracy': 0.25,
         'error': 0.75,
         'num_correct_predictions': 4,
         'num_incorrect_predictions': 12,
         'num_predictions': 16}

        ```
        """
        self._tracker.update(correct.detach())

    def value(self, prefix: str = "", suffix: str = "") -> dict[str, int | float]:
        tracker = self._tracker.all_reduce()
        if not tracker.count:
            msg = f"{self.__class__.__qualname__} is empty"
            raise EmptyMetricError(msg)

        accuracy = tracker.mean()
        num_correct_predictions = int(tracker.sum())
        num_predictions = tracker.count
        results = {
            f"{prefix}accuracy{suffix}": accuracy,
            f"{prefix}error{suffix}": 1.0 - accuracy,
        }
        if self._track_num_predictions:
            results.update(
                {
                    f"{prefix}num_correct_predictions{suffix}": num_correct_predictions,
                    f"{prefix}num_incorrect_predictions{suffix}": num_predictions
                    - num_correct_predictions,
                    f"{prefix}num_predictions{suffix}": num_predictions,
                }
            )
        return results
