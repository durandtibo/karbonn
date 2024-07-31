r"""Contain trackers for tensors."""

from __future__ import annotations

__all__ = ["MeanTensorTracker", "ExtremaTensorTracker"]

from typing import TYPE_CHECKING, Any

import torch

from karbonn.distributed.ddp import MAX, MIN, SUM, sync_reduce
from karbonn.utils.tracker.exception import EmptyTrackerError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor


class MeanTensorTracker:
    r"""Implement a tracker to compute the mean value of
    ``torch.Tensor``s.

    The mean value is updated by keeping local variables ``total``
    and ``count``. ``count`` tracks the number of values, and
    ``total`` tracks the sum of the values.
    This tracker has a constant space complexity.

    Args:
        count: The initial count value.
        total: The initial total value.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.utils.tracker import MeanTensorTracker
    >>> tracker = MeanTensorTracker()
    >>> tracker.update(torch.arange(6))
    >>> tracker.update(torch.tensor([4.0, 1.0]))
    >>> tracker.mean()
    2.5
    >>> tracker.sum()
    20.0
    >>> tracker.count
    8

    ```
    """

    def __init__(self, count: int = 0, total: float = 0.0) -> None:
        self._count = int(count)
        self._total = total

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(count={self._count:,}, total={self._total})"

    @property
    def count(self) -> int:
        r"""The number of predictions in the tracker."""
        return self._count

    @property
    def total(self) -> float:
        r"""The total sum value in the tracker."""
        return self._total

    def reset(self) -> None:
        r"""Reset the tracker.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.tracker import MeanTensorTracker
        >>> tracker = MeanTensorTracker(count=6, total=42.0)
        >>> tracker.reset()
        >>> tracker.count
        0
        >>> tracker.total
        0.0

        ```
        """
        self._count = 0
        self._total = 0.0

    def update(self, tensor: Tensor) -> None:
        r"""Update the tracker given a new tensor.

        Args:
            tensor: The tensor to add to the tracker.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.tracker import MeanTensorTracker
        >>> tracker = MeanTensorTracker()
        >>> tracker.update(torch.arange(6))
        >>> tracker.count
        6

        ```
        """
        self._total += tensor.sum().item()
        self._count += tensor.numel()

    def average(self) -> float:
        r"""Compute the average value.

        Returns:
            The average value.

        Raises:
            EmptyTrackerError: if the tracker is empty.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.tracker import MeanTensorTracker
        >>> tracker = MeanTensorTracker()
        >>> tracker.update(torch.arange(6))
        >>> tracker.average()
        2.5

        ```
        """
        return self.mean()

    def mean(self) -> float:
        r"""Get the mean value.

        Returns:
            The mean value.

        Raises:
            EmptyTrackerError: if the tracker is empty.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.tracker import MeanTensorTracker
        >>> tracker = MeanTensorTracker()
        >>> tracker.update(torch.arange(6))
        >>> tracker.mean()
        2.5

        ```
        """
        if not self._count:
            msg = "The tracker is empty"
            raise EmptyTrackerError(msg)
        return float(self._total) / float(self._count)

    def sum(self) -> float:
        r"""Get the sum of all the values.

        Returns:
            The sum of all the values.

        Raises:
            EmptyTrackerError: if the tracker is empty.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.tracker import MeanTensorTracker
        >>> tracker = MeanTensorTracker()
        >>> tracker.update(torch.arange(6))
        >>> tracker.sum()
        15.0

        ```
        """
        if not self._count:
            msg = "The tracker is empty"
            raise EmptyTrackerError(msg)
        return self._total

    def all_reduce(self) -> MeanTensorTracker:
        r"""Reduce the tracker values across all machines in such a way
        that all get the final result.

        The sum value is reduced by summing all the sum values (1 sum
        value per distributed process). The count value is reduced by
        summing all the count values (1 count value per distributed
        process).

        In a non-distributed setting, this method returns a copy of
        the current tracker.

        Returns:
            The reduced tracker.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.tracker import MeanTensorTracker
        >>> tracker = MeanTensorTracker()
        >>> tracker.update(torch.arange(6))
        >>> reduced_tracker = tracker.all_reduce()

        ```
        """
        return MeanTensorTracker(
            count=int(sync_reduce(self._count, SUM)), total=sync_reduce(self._total, SUM)
        )

    def clone(self) -> MeanTensorTracker:
        r"""Create a copy of the current tracker.

        Returns:
            A copy of the current tracker.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.tracker import MeanTensorTracker
        >>> tracker = MeanTensorTracker()
        >>> tracker.update(torch.arange(6))
        >>> tracker_cloned = tracker.clone()
        >>> tracker.update(torch.ones(3))
        >>> tracker.sum()
        18.0
        >>> tracker_cloned.sum()
        15.0

        ```
        """
        return MeanTensorTracker(count=self._count, total=self._total)

    def equal(self, other: Any) -> bool:
        r"""Indicate if two trackers are equal or not.

        Args:
            other: The value to compare.

        Returns:
            ``True`` if the trackers are equal,
                ``False`` otherwise.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.tracker import MeanTensorTracker
        >>> tracker1 = MeanTensorTracker()
        >>> tracker1.update(torch.arange(6))
        >>> tracker2 = MeanTensorTracker()
        >>> tracker2.update(torch.ones(3))
        >>> tracker1.equal(tracker2)
        False

        ```
        """
        if not isinstance(other, MeanTensorTracker):
            return False
        return self.state_dict() == other.state_dict()

    def merge(self, trackers: Iterable[MeanTensorTracker]) -> MeanTensorTracker:
        r"""Merge several trackers with the current tracker and returns a
        new tracker.

        Args:
            trackers: The trackers to merge to the current tracker.

        Returns:
            The merged tracker.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.tracker import MeanTensorTracker
        >>> tracker1 = MeanTensorTracker()
        >>> tracker1.update(torch.arange(6))
        >>> tracker2 = MeanTensorTracker()
        >>> tracker2.update(torch.ones(3))
        >>> tracker3 = tracker1.merge([tracker2])
        >>> tracker3.sum()
        18.0

        ```
        """
        count, total = self.count, self.total
        for tracker in trackers:
            count += tracker.count
            total += tracker.total
        return MeanTensorTracker(total=total, count=count)

    def merge_(self, trackers: Iterable[MeanTensorTracker]) -> None:
        r"""Merge several trackers into the current tracker.

        In-place version of ``merge``.

        Args:
            trackers: The trackers to merge to the current tracker.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.tracker import MeanTensorTracker
        >>> tracker1 = MeanTensorTracker()
        >>> tracker1.update(torch.arange(6))
        >>> tracker2 = MeanTensorTracker()
        >>> tracker2.update(torch.ones(3))
        >>> tracker1.merge_([tracker2])
        >>> tracker1.sum()
        18.0

        ```
        """
        for tracker in trackers:
            self._count += tracker.count
            self._total += tracker.total

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Load a state to the history tracker.

        Args:
            state_dict: A dictionary containing state keys with values.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.tracker import MeanTensorTracker
        >>> tracker = MeanTensorTracker()
        >>> tracker.load_state_dict({"count": 6, "total": 42.0})
        >>> tracker.count
        6
        >>> tracker.sum()
        42.0

        ```
        """
        self._count = int(state_dict["count"])
        self._total = float(state_dict["total"])

    def state_dict(self) -> dict[str, int | float]:
        r"""Return a dictionary containing state values.

        Returns:
            The state values in a dict.

        Example usage:

        ```pycon

        >>> from karbonn.utils.tracker import MeanTensorTracker
        >>> tracker = MeanTensorTracker(count=6, total=42.0)
        >>> tracker.state_dict()
        {'count': 6, 'total': 42.0}

        ```
        """
        return {"count": self._count, "total": self._total}


class ExtremaTensorTracker:
    r"""Implement a tracker to compute the minimum and maximum values of
    ``torch.Tensor``s.

    The mean value is updated by keeping local variables ``min_value``
    and ``max_value``. ``min_value`` tracks the minimum value, and
    ``max_value`` tracks the maximum value.
    This tracker has a constant space complexity.

    Args:
        count: The initial count value.
        min_value: The initial minimum value.
        max_value: The initial maximum value.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.utils.tracker import ExtremaTensorTracker
    >>> tracker = ExtremaTensorTracker()
    >>> tracker.update(torch.arange(6))
    >>> tracker.update(torch.tensor([4.0, 1.0]))
    >>> tracker.max()
    5.0
    >>> tracker.min()
    0.0

    ```
    """

    def __init__(
        self, count: int = 0, min_value: float = float("inf"), max_value: float = float("-inf")
    ) -> None:
        self._count = int(count)
        self._min_value = float(min_value)
        self._max_value = float(max_value)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(count={self._count:,}, "
            f"min_value={self._min_value}, max_value={self._max_value})"
        )

    @property
    def count(self) -> int:
        r"""The number of predictions in the tracker."""
        return self._count

    def reset(self) -> None:
        r"""Reset the tracker.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.tracker import ExtremaTensorTracker
        >>> tracker = ExtremaTensorTracker(count=6, min_value=-2.0, max_value=6.0)
        >>> tracker.reset()
        >>> tracker.count
        0

        ```
        """
        self._count = 0
        self._max_value = float("-inf")
        self._min_value = float("inf")

    def update(self, tensor: Tensor) -> None:
        r"""Update the tracker given a new tensor.

        Args:
            tensor: The tensor to add to the tracker.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.tracker import ExtremaTensorTracker
        >>> tracker = ExtremaTensorTracker()
        >>> tracker.update(torch.arange(6))
        >>> tracker.max()
        5.0
        >>> tracker.min()
        0.0

        ```
        """
        min_value, max_value = torch.aminmax(tensor)
        self._max_value = max(self._max_value, max_value.item())
        self._min_value = min(self._min_value, min_value.item())
        self._count += tensor.numel()

    def max(self) -> float:
        r"""Get the max value.

        Returns:
            The max value.

        Raises:
            EmptyTrackerError:if the tracker is empty.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.tracker import ExtremaTensorTracker
        >>> tracker = ExtremaTensorTracker()
        >>> tracker.update(torch.arange(6))
        >>> tracker.max()
        5.0

        ```
        """
        if not self._count:
            msg = "The tracker is empty"
            raise EmptyTrackerError(msg)
        return float(self._max_value)

    def min(self) -> float:
        r"""Get the min value.

        Returns:
            The min value.

        Raises:
            EmptyTrackerError: if the tracker is empty.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.tracker import ExtremaTensorTracker
        >>> tracker = ExtremaTensorTracker()
        >>> tracker.update(torch.arange(6))
        >>> tracker.min()
        0.0

        ```
        """
        if not self._count:
            msg = "The tracker is empty"
            raise EmptyTrackerError(msg)
        return float(self._min_value)

    def all_reduce(self) -> ExtremaTensorTracker:
        r"""Reduce the tracker values across all machines in such a way
        that all get the final result.

        The maximum value is reduced by computing the maximum between
        the maximum values (1 maximum value per distributed process).
        The minimum value is reduced by computing the minimum between
        the minimum values (1 minimum value per distributed process).

        Returns:
            The reduced tracker.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.tracker import ExtremaTensorTracker
        >>> tracker = ExtremaTensorTracker()
        >>> tracker.update(torch.arange(6))
        >>> reduced_tracker = tracker.all_reduce()

        ```
        """
        return ExtremaTensorTracker(
            count=int(sync_reduce(self._count, SUM)),
            min_value=sync_reduce(self._min_value, MIN),
            max_value=sync_reduce(self._max_value, MAX),
        )

    def clone(self) -> ExtremaTensorTracker:
        r"""Create a copy of the current tracker.

        Returns:
            A copy of the current tracker.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.tracker import ExtremaTensorTracker
        >>> tracker = ExtremaTensorTracker()
        >>> tracker.update(torch.arange(6))
        >>> tracker_cloned = tracker.clone()
        >>> tracker.update(torch.ones(3))
        >>> tracker.count
        9
        >>> tracker_cloned.count
        6

        ```
        """
        return ExtremaTensorTracker(
            count=self._count, min_value=self._min_value, max_value=self._max_value
        )

    def equal(self, other: Any) -> bool:
        r"""Indicate if two trackers are equal or not.

        Args:
            other: The value to compare.

        Returns:
            ``True`` if the trackers are equal,
                ``False`` otherwise.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.tracker import ExtremaTensorTracker
        >>> tracker1 = ExtremaTensorTracker()
        >>> tracker1.update(torch.arange(6))
        >>> tracker2 = ExtremaTensorTracker()
        >>> tracker2.update(torch.ones(3))
        >>> tracker1.equal(tracker2)
        False

        ```
        """
        if not isinstance(other, ExtremaTensorTracker):
            return False
        return self.state_dict() == other.state_dict()

    def merge(self, trackers: Iterable[ExtremaTensorTracker]) -> ExtremaTensorTracker:
        r"""Merge several trackers with the current tracker and returns a
        new tracker.

        Args:
            trackers: The trackers to merge to the current tracker.

        Returns:
            The merged tracker.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.tracker import ExtremaTensorTracker
        >>> tracker1 = ExtremaTensorTracker()
        >>> tracker1.update(torch.arange(6) + 3)
        >>> tracker2 = ExtremaTensorTracker()
        >>> tracker2.update(torch.ones(3))
        >>> tracker3 = tracker1.merge([tracker2])
        >>> tracker3.count
        9
        >>> tracker3.max()
        8.0
        >>> tracker3.min()
        1.0

        ```
        """
        count, min_value, max_value = self._count, self._min_value, self._max_value
        for tracker in trackers:
            count += tracker.count
            min_value = min(min_value, tracker._min_value)
            max_value = max(max_value, tracker._max_value)
        return ExtremaTensorTracker(count=count, min_value=min_value, max_value=max_value)

    def merge_(self, trackers: Iterable[ExtremaTensorTracker]) -> None:
        r"""Merge several trackers into the current tracker.

        In-place version of ``merge``.

        Args:
            trackers: The trackers to merge to the current tracker.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.tracker import ExtremaTensorTracker
        >>> tracker1 = ExtremaTensorTracker()
        >>> tracker1.update(torch.arange(6) + 3)
        >>> tracker2 = ExtremaTensorTracker()
        >>> tracker2.update(torch.ones(3))
        >>> tracker1.merge_([tracker2])
        >>> tracker1.count
        9
        >>> tracker1.max()
        8.0
        >>> tracker1.min()
        1.0

        ```
        """
        for tracker in trackers:
            self._count += tracker.count
            self._min_value = min(self._min_value, tracker._min_value)
            self._max_value = max(self._max_value, tracker._max_value)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""Load a state to the history tracker.

        Args:
            state_dict: A dictionary containing state keys with values.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.tracker import ExtremaTensorTracker
        >>> tracker = ExtremaTensorTracker()
        >>> tracker.load_state_dict({"count": 6, "max_value": 42.0, "min_value": -2.0})
        >>> tracker.count
        6
        >>> tracker.min()
        -2.0
        >>> tracker.max()
        42.0

        ```
        """
        self._count = int(state_dict["count"])
        self._min_value = float(state_dict["min_value"])
        self._max_value = float(state_dict["max_value"])

    def state_dict(self) -> dict[str, Any]:
        r"""Return a dictionary containing state values.

        Returns:
            The state values in a dict.

        Example usage:

        ```pycon

        >>> from karbonn.utils.tracker import ExtremaTensorTracker
        >>> tracker = ExtremaTensorTracker(count=6, max_value=42.0, min_value=-2.0)
        >>> tracker.state_dict()
        {'count': 6, 'max_value': 42.0, 'min_value': -2.0}

        ```
        """
        return {"count": self._count, "max_value": self._max_value, "min_value": self._min_value}
