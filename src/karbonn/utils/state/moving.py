r"""Implement states to track the moving average value of float
number."""

from __future__ import annotations

__all__ = ["ExponentialMovingAverageState", "MovingAverageState"]

from collections import deque
from typing import TYPE_CHECKING, Any

import torch

from karbonn.utils.state.exception import EmptyStateError

if TYPE_CHECKING:
    from collections.abc import Iterable


class MovingAverageState:
    r"""Implement a state to track the moving average value of float
    number.

    Args:
        values: The initial values.
        window_size: The maximum window size.

    Example usage:

    ```pycon

    >>> from karbonn.utils.state import MovingAverageState
    >>> state = MovingAverageState()
    >>> for i in range(11):
    ...     state.update(i)
    ...
    >>> state.smoothed_average()
    5.0

    ```
    """

    def __init__(self, values: Iterable[float] = (), window_size: int = 20) -> None:
        self._deque = deque(values, maxlen=window_size)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(window_size={self.window_size:,})"

    @property
    def values(self) -> tuple[float, ...]:
        r"""The values in the moving average window."""
        return tuple(self._deque)

    @property
    def window_size(self) -> int:
        r"""The moving average window size."""
        return self._deque.maxlen

    def clone(self) -> MovingAverageState:
        r"""Return a copy of the current state.

        Returns:
            A copy of the current state.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.state import MovingAverageState
        >>> state = MovingAverageState(values=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
        >>> state_cloned = state.clone()
        >>> state.update(11)
        >>> state.update(12)
        >>> state.smoothed_average()
        6.0
        >>> state_cloned.smoothed_average()
        5.0

        ```
        """
        return MovingAverageState(values=tuple(self._deque), window_size=self.window_size)

    def equal(self, other: Any) -> bool:
        r"""Indicate if two states are equal or not.

        Args:
            other: The value to compare.

        Returns:
            ``True`` if the states are equal, ``False`` otherwise.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.state import MovingAverageState
        >>> state1 = MovingAverageState(values=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
        >>> state2 = MovingAverageState(values=(1.0, 1.0, 1.0))
        >>> state1.equal(state2)
        False

        ```
        """
        if not isinstance(other, MovingAverageState):
            return False
        return self.state_dict() == other.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""Load a state to the state.

        Args:
            state_dict: A dictionary containing state keys with values.

        Example usage:

        ```pycon

        >>> from karbonn.utils.state import MovingAverageState
        >>> state = MovingAverageState()
        >>> state.load_state_dict({"values": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), "window_size": 20})
        >>> state.smoothed_average()
        5.0

        ```
        """
        self._deque = deque(state_dict["values"], maxlen=state_dict["window_size"])

    def reset(self) -> None:
        r"""Reset the state.

        Example usage:

        ```pycon

        >>> from karbonn.utils.state import MovingAverageState
        >>> state = MovingAverageState()
        >>> for i in range(11):
        ...     state.update(i)
        ...
        >>> state.reset()
        >>> state.values
        ()

        ```
        """
        self._deque.clear()

    def smoothed_average(self) -> float:
        r"""Compute the smoothed average value.

        Returns:
            The smoothed average value.

        Raises:
            EmptyStateError: if the state is empty.

        Example usage:

        ```pycon

        >>> from karbonn.utils.state import MovingAverageState
        >>> state = MovingAverageState()
        >>> for i in range(11):
        ...     state.update(i)
        ...
        >>> state.smoothed_average()
        5.0

        ```
        """
        if not self._deque:
            msg = "The state is empty"
            raise EmptyStateError(msg)
        return torch.as_tensor(self.values).float().mean().item()

    def state_dict(self) -> dict[str, Any]:
        r"""Return a dictionary containing state values.

        Returns:
            The state values in a dict.

        Example usage:

        ```pycon

        >>> from karbonn.utils.state import MovingAverageState
        >>> state = MovingAverageState()
        >>> for i in range(11):
        ...     state.update(i)
        ...
        >>> state.state_dict()
        {'values': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'window_size': 20}

        ```
        """
        return {"values": self.values, "window_size": self.window_size}

    def update(self, value: float) -> None:
        r"""Update the state given a new value.

        Args:
            value: The value to add to the state.

        Example usage:

        ```pycon

        >>> from karbonn.utils.state import MovingAverageState
        >>> state = MovingAverageState()
        >>> for i in range(11):
        ...     state.update(i)
        ...
        >>> state.smoothed_average()
        5.0

        ```
        """
        self._deque.append(value)


class ExponentialMovingAverageState:
    r"""Implement a state to track the exponential moving average value
    of float number.

    Args:
        alpha: The smoothing factor such as ``0 < alpha < 1``.
        count: The initial count value.
        smoothed_average: The initial value of the smoothed average.

    Example usage:

    ```pycon

    >>> from karbonn.utils.state import ExponentialMovingAverageState
    >>> state = ExponentialMovingAverageState()
    >>> for i in range(11):
    ...     state.update(i)
    ...
    >>> state.count
    11.0
    >>> state.smoothed_average()
    1.036567...

    ```
    """

    def __init__(
        self,
        alpha: float = 0.98,
        count: float = 0,
        smoothed_average: float = 0.0,
    ) -> None:
        self._alpha = float(alpha)
        self._count = float(count)
        self._smoothed_average = float(smoothed_average)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(alpha={self._alpha}, count={self._count:,}, "
            f"smoothed_average={self._smoothed_average}, )"
        )

    @property
    def count(self) -> float:
        r"""The number of examples in the state since the last reset."""
        return self._count

    def clone(self) -> ExponentialMovingAverageState:
        r"""Return a copy of the current state.

        Returns:
            A copy of the current state.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.state import ExponentialMovingAverageState
        >>> state = ExponentialMovingAverageState(smoothed_average=42.0, count=11)
        >>> state_cloned = state.clone()
        >>> state.update(1)
        >>> state.smoothed_average()
        41.18
        >>> state.count
        12.0
        >>> state_cloned.smoothed_average()
        42.0
        >>> state_cloned.count
        11.0

        ```
        """
        return ExponentialMovingAverageState(
            alpha=self._alpha,
            count=self._count,
            smoothed_average=self._smoothed_average,
        )

    def equal(self, other: Any) -> bool:
        r"""Indicate if two states are equal or not.

        Args:
            other: The value to compare.

        Returns:
            ``True`` if the states are equal, ``False`` otherwise.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.state import ExponentialMovingAverageState
        >>> state1 = ExponentialMovingAverageState(count=10, smoothed_average=42.0)
        >>> state2 = ExponentialMovingAverageState()
        >>> state1.equal(state2)
        False

        ```
        """
        if not isinstance(other, ExponentialMovingAverageState):
            return False
        return self.state_dict() == other.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""Load a state to the history tracker.

        Args:
            state_dict: A dictionary containing state keys with values.

        Example usage:

        ```pycon

        >>> from karbonn.utils.state import ExponentialMovingAverageState
        >>> state = ExponentialMovingAverageState()
        >>> for i in range(11):
        ...     state.update(i)
        ...
        >>> state.load_state_dict({"alpha": 0.98, "count": 11, "smoothed_average": 42.0})
        >>> state.count
        11.0
        >>> state.smoothed_average()
        42.0

        ```
        """
        self._alpha = float(state_dict["alpha"])
        self._count = float(state_dict["count"])
        self._smoothed_average = float(state_dict["smoothed_average"])

    def reset(self) -> None:
        r"""Reset the state.

        Example usage:

        ```pycon

        >>> from karbonn.utils.state import ExponentialMovingAverageState
        >>> state = ExponentialMovingAverageState()
        >>> for i in range(11):
        ...     state.update(i)
        ...
        >>> state.reset()
        >>> state.count
        0.0

        ```
        """
        self._count = 0.0
        self._smoothed_average = 0.0

    def smoothed_average(self) -> float:
        r"""Compute the smoothed average value.

        Returns:
            The smoothed average value.

        Raises:
            EmptyStateError: if the state is empty.

        Example usage:

        ```pycon

        >>> from karbonn.utils.state import ExponentialMovingAverageState
        >>> state = ExponentialMovingAverageState()
        >>> for i in range(11):
        ...     state.update(i)
        ...
        >>> state.smoothed_average()
        1.036567...

        ```
        """
        if not self._count:
            msg = "The state is empty"
            raise EmptyStateError(msg)
        return self._smoothed_average

    def state_dict(self) -> dict[str, Any]:
        r"""Return a dictionary containing state values.

        Returns:
            The state values in a dict.

        Example usage:

        ```pycon

        >>> from karbonn.utils.state import ExponentialMovingAverageState
        >>> state = ExponentialMovingAverageState()
        >>> for i in range(11):
        ...     state.update(i)
        ...
        >>> state.state_dict()
        {'alpha': 0.98, 'count': 11.0, 'smoothed_average': 1.036567...}

        ```
        """
        return {
            "alpha": self._alpha,
            "count": self._count,
            "smoothed_average": self._smoothed_average,
        }

    def update(self, value: float) -> None:
        r"""Update the state given a new value.

        Args:
            value: The value to add to the state.

        Example usage:

        ```pycon

        >>> from karbonn.utils.state import ExponentialMovingAverageState
        >>> state = ExponentialMovingAverageState()
        >>> for i in range(11):
        ...     state.update(i)
        ...
        >>> state.count
        11.0

        ```
        """
        self._smoothed_average = self._alpha * self._smoothed_average + (1.0 - self._alpha) * value
        self._count += 1
