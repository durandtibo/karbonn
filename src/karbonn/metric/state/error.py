r"""Contain the error-based metric states."""

from __future__ import annotations

__all__ = ["ErrorState", "MeanErrorState"]

from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping
from minrecord import BaseRecord, MinScalarRecord

from karbonn.metric import EmptyMetricError
from karbonn.metric.state.base import BaseState
from karbonn.utils.tracker import MeanTensorTracker, ScalableTensorTracker

if TYPE_CHECKING:
    import torch


class ErrorState(BaseState):
    r"""Implements a metric state to capture some metrics about the
    errors.

    This state has a constant space complexity.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.metric.state import ErrorState
    >>> state = ErrorState()
    >>> state
    ErrorState(
      (meter): ScalableTensorTracker(count=0, total=0.0, min_value=inf, max_value=-inf)
      (track_num_predictions): True
    )
    >>> state.get_records("error_")
    (MinScalarRecord(name=error_mean, max_size=10, size=0),
     MinScalarRecord(name=error_min, max_size=10, size=0),
     MinScalarRecord(name=error_max, max_size=10, size=0),
     MinScalarRecord(name=error_sum, max_size=10, size=0))
    >>> state.update(torch.arange(6))
    >>> state.value("error_")
    {'error_mean': 2.5,
     'error_min': 0.0,
     'error_max': 5.0,
     'error_sum': 15.0,
     'error_num_predictions': 6}

    ```
    """

    def __init__(self, track_num_predictions: bool = True) -> None:
        self._meter = ScalableTensorTracker()
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
                {
                    "meter": self._meter,
                    "num_predictions": self.num_predictions,
                    "track_num_predictions": self._track_num_predictions,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    @property
    def num_predictions(self) -> int:
        return self._meter.count

    def get_records(self, prefix: str = "", suffix: str = "") -> tuple[BaseRecord, ...]:
        return (
            MinScalarRecord(name=f"{prefix}mean{suffix}"),
            MinScalarRecord(name=f"{prefix}min{suffix}"),
            MinScalarRecord(name=f"{prefix}max{suffix}"),
            MinScalarRecord(name=f"{prefix}sum{suffix}"),
        )

    def reset(self) -> None:
        self._meter.reset()

    def update(self, error: torch.Tensor) -> None:
        r"""Update the metric state with a new tensor of errors.

        Args:
            error: A tensor of errors.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.metric.state import ErrorState
        >>> state = ErrorState()
        >>> state.update(torch.arange(6))
        >>> state.value("error_")
        {'error_mean': 2.5,
         'error_min': 0.0,
         'error_max': 5.0,
         'error_sum': 15.0,
         'error_num_predictions': 6}

        ```
        """
        self._meter.update(error.detach())

    def value(self, prefix: str = "", suffix: str = "") -> dict[str, int | float]:
        meter = self._meter.all_reduce()
        if not meter.count:
            msg = f"{self.__class__.__qualname__} is empty"
            raise EmptyMetricError(msg)

        results = {
            f"{prefix}mean{suffix}": self._meter.mean(),
            f"{prefix}min{suffix}": self._meter.min(),
            f"{prefix}max{suffix}": self._meter.max(),
            f"{prefix}sum{suffix}": self._meter.sum(),
        }
        if self._track_num_predictions:
            results[f"{prefix}num_predictions{suffix}"] = meter.count
        return results


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
                {
                    "meter": self._meter,
                    "num_predictions": self.num_predictions,
                    "track_num_predictions": self._track_num_predictions,
                }
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
