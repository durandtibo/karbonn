r"""Contain the error-based metric states."""

from __future__ import annotations

__all__ = ["MeanErrorState"]

from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping
from minrecord import BaseRecord, MinScalarRecord

from karbonn.metric import EmptyMetricError
from karbonn.metric.state.base import BaseState
from karbonn.utils.tracker import MeanTensorTracker

if TYPE_CHECKING:
    import torch


class MeanErrorState(BaseState):
    r"""Implement a metric state to capture the mean error value.

    This state has a constant space complexity.

    Args:
        track_num_predictions: If ``True``, the state tracks and
            returns the number of predictions.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.metric.state import MeanErrorState
    >>> state = MeanErrorState()
    >>> state
    MeanErrorState(
      (meter): MeanTensorTracker(count=0, total=0.0)
      (track_num_predictions): True
    )
    >>> state.get_records("error_")
    (MinScalarRecord(name=error_mean, max_size=10, size=0),)
    >>> state.update(torch.arange(6))
    >>> state.value("error_")
    {'error_mean': 2.5, 'error_num_predictions': 6}

    ```
    """

    def __init__(self, track_num_predictions: bool = True) -> None:
        self._meter = MeanTensorTracker()
        self._track_num_predictions = bool(track_num_predictions)

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping(
                {"meter": self._meter, "track_num_predictions": self._track_num_predictions}
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(
            str_mapping(
                {"meter": self._meter, "track_num_predictions": self._track_num_predictions}
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    @property
    def num_predictions(self) -> int:
        return self._meter.count

    def get_records(self, prefix: str = "", suffix: str = "") -> tuple[BaseRecord, ...]:
        return (MinScalarRecord(name=f"{prefix}mean{suffix}"),)

    def reset(self) -> None:
        self._meter.reset()

    def update(self, error: torch.Tensor) -> None:
        r"""Update the metric state with a new tensor of errors.

        Args:
            error: A tensor of errors.


        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.metric.state import MeanErrorState
        >>> state = MeanErrorState()
        >>> state.update(torch.arange(6))
        >>> state.value("error_")
        {'error_mean': 2.5, 'error_num_predictions': 6}

        ```
        """
        self._meter.update(error.detach())

    def value(self, prefix: str = "", suffix: str = "") -> dict[str, int | float]:
        meter = self._meter.all_reduce()
        if not meter.count:
            msg = f"{self.__class__.__qualname__} is empty"
            raise EmptyMetricError(msg)

        results = {f"{prefix}mean{suffix}": self._meter.mean()}
        if self._track_num_predictions:
            results[f"{prefix}num_predictions{suffix}"] = meter.count
        return results
