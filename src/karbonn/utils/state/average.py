r"""Implement a state to track the average value of float number."""

from __future__ import annotations

__all__ = ["AverageState"]

from typing import TYPE_CHECKING, Any

from coola.utils import str_indent, str_mapping

from karbonn.distributed.ddp import SUM, sync_reduce
from karbonn.utils.state.exception import EmptyStateError

if TYPE_CHECKING:
    from collections.abc import Iterable


class AverageState:
    r"""Implement a state to track the average value of float number.

    Args:
        total: The initial total value.
        count: The initial count value.

    Example usage:

    ```pycon

    >>> from karbonn.utils.state import AverageState
    >>> state = AverageState()
    >>> for i in range(11):
    ...     state.update(i)
    ...
    >>> state.average()
    5.0
    >>> state.sum()
    55.0
    >>> state.count
    11.0

    ```
    """

    def __init__(self, total: float = 0.0, count: float = 0) -> None:
        self._total = float(total)
        self._count = float(count)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(count={self._count:,}, total={self._total})"

    def __str__(self) -> str:
        stats = str_indent(
            str_mapping(
                {
                    "average": self.average() if self.count else "N/A (empty)",
                    "count": self.count,
                    "total": self.total,
                },
            )
        )
        return f"{self.__class__.__qualname__}(\n  {stats}\n)"

    @property
    def count(self) -> float:
        r"""The number of examples in the state since the last reset."""
        return self._count

    @property
    def total(self) -> float:
        r"""The total of the values added to the state since the last
        reset."""
        return self._total

    def all_reduce(self) -> AverageState:
        r"""Reduce the state values across all machines in such a way
        that all get the final result.

        The total value is reduced by summing all the sum values
        (1 total value per distributed process).
        The count value is reduced by summing all the count values
        (1 count value per distributed process).

        Returns:
            The reduced state.

        Example usage:

        ```pycon

        >>> from karbonn.utils.state import AverageState
        >>> state = AverageState()
        >>> state.update(6)
        >>> reduced_meter = state.all_reduce()

        ```
        """
        return AverageState(
            total=sync_reduce(self._total, SUM),
            count=sync_reduce(self._count, SUM),
        )

    def average(self) -> float:
        r"""Return the average value.

        Returns:
            The average value.

        Raises:
            EmptyStateError: if the state is empty.

        Example usage:

        ```pycon

        >>> from karbonn.utils.state import AverageState
        >>> state = AverageState()
        >>> for i in range(11):
        ...     state.update(i)
        ...
        >>> state.average()
        5.0

        ```
        """
        if not self._count:
            msg = "The state is empty"
            raise EmptyStateError(msg)
        return self._total / self._count

    def clone(self) -> AverageState:
        r"""Return a copy of the current state.

        Returns:
            A copy of the current state.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.state import AverageState
        >>> state = AverageState(total=55.0, count=11)
        >>> meter_cloned = state.clone()
        >>> state.update(1)
        >>> state.sum()
        56.0
        >>> state.count
        12.0
        >>> meter_cloned.sum()
        55.0
        >>> meter_cloned.count
        11.0

        ```
        """
        return AverageState(total=self.total, count=self.count)

    def equal(self, other: Any) -> bool:
        r"""Indicate if two states are equal or not.

        Args:
            other: The value to compare.

        Returns:
            ``True`` if the states are equal, ``False`` otherwise.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.state import AverageState
        >>> meter1 = AverageState(total=55.0, count=11)
        >>> meter2 = AverageState(total=3.0, count=3)
        >>> meter1.equal(meter2)
        False

        ```
        """
        if not isinstance(other, AverageState):
            return False
        return self.state_dict() == other.state_dict()

    def merge(self, states: Iterable[AverageState]) -> AverageState:
        r"""Merge several states with the current state and return a new
        state.

        Args:
            states: The states to merge to the current state.

        Returns:
            The merged state.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.state import AverageState
        >>> meter1 = AverageState(total=55.0, count=10)
        >>> meter2 = AverageState(total=3.0, count=3)
        >>> meter3 = meter1.merge([meter2])
        >>> meter3.count
        13.0
        >>> meter3.sum()
        58.0

        ```
        """
        count, total = self.count, self.total
        for meter in states:
            count += meter.count
            total += meter.total
        return AverageState(total=total, count=count)

    def merge_(self, states: Iterable[AverageState]) -> None:
        r"""Merge several states into the current state.

        In-place version of ``merge``.

        Args:
            states: The states to merge to the current state.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.state import AverageState
        >>> meter1 = AverageState(total=55.0, count=10)
        >>> meter2 = AverageState(total=3.0, count=3)
        >>> meter1.merge_([meter2])
        >>> meter1.count
        13.0
        >>> meter1.sum()
        58.0

        ```
        """
        for meter in states:
            self._count += meter.count
            self._total += meter.total

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""Load a state to the history tracker.

        Args:
            state_dict: A dictionary containing state keys with values.

        Example usage:

        ```pycon

        >>> from karbonn.utils.state import AverageState
        >>> state = AverageState()
        >>> state.load_state_dict({"count": 11.0, "total": 55.0})
        >>> state.count
        11.0
        >>> state.sum()
        55.0

        ```
        """
        self._total = float(state_dict["total"])
        self._count = float(state_dict["count"])

    def reset(self) -> None:
        r"""Reset the state.

        Example usage:

        ```pycon

        >>> from karbonn.utils.state import AverageState
        >>> state = AverageState()
        >>> for i in range(11):
        ...     state.update(i)
        ...
        >>> state.reset()
        >>> state.count
        0.0

        ```
        """
        self._total = 0.0
        self._count = 0.0

    def state_dict(self) -> dict[str, Any]:
        r"""Return a dictionary containing state values.

        Returns:
            The state values in a dict.

        Example usage:

        ```pycon

        >>> from karbonn.utils.state import AverageState
        >>> state = AverageState()
        >>> for i in range(11):
        ...     state.update(i)
        ...
        >>> state.state_dict()
        {'count': 11.0, 'total': 55.0}

        ```
        """
        return {"count": self._count, "total": self._total}

    def sum(self) -> float:
        r"""Return the sum value.

        Returns:
            The sum value.

        Raises:
            EmptyStateError: if the state is empty.

        Example usage:

        ```pycon

        >>> from karbonn.utils.state import AverageState
        >>> state = AverageState()
        >>> for i in range(11):
        ...     state.update(i)
        ...
        >>> state.sum()
        55.0

        ```
        """
        if not self._count:
            msg = "The state is empty"
            raise EmptyStateError(msg)
        return self._total

    def update(self, value: float, num_examples: float = 1) -> None:
        r"""Update the state given a new value and the number of
        examples.

        Args:
            value: The value to add to the state.
            num_examples: The number of examples. This argument is
                mainly used to deal with mini-batches of different
                sizes.

        Example usage:

        ```pycon

        >>> from karbonn.utils.state import AverageState
        >>> state = AverageState()
        >>> for i in range(11):
        ...     state.update(i)
        ...
        >>> state.sum()
        55.0

        ```
        """
        num_examples = float(num_examples)
        self._total += float(value) * num_examples
        self._count += num_examples
