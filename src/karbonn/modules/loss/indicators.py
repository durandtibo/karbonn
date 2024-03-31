r"""Contain relative indicator functions."""

from __future__ import annotations

__all__ = [
    "BaseRelativeIndicator",
    "ArithmeticalMeanIndicator",
    "ClassicalRelativeIndicator",
    "ReversedRelativeIndicator",
]


import torch
from torch import nn

from karbonn.functional.loss.relative import (
    arithmetical_mean_indicator,
    classical_relative_indicator,
    reversed_relative_indicator,
)


class BaseRelativeIndicator(nn.Module):
    r"""Define the base class to implement a relative indicator function.

    The indicators are designed based on
    https://en.wikipedia.org/wiki/Relative_change#Indicators_of_relative_change.
    """

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Return the indicator values.

        Args:
            prediction: The predictions.
            target: The target values.

        Returns:
            The indicator values.
        """


class ArithmeticalMeanIndicator(BaseRelativeIndicator):
    r"""Implement the arithmetical mean change indicator function.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.modules.loss import ArithmeticalMeanIndicator
    >>> prediction = torch.randn(3, 5, requires_grad=True)
    >>> target = torch.randn(3, 5)
    >>> indicator = ArithmeticalMeanIndicator()
    >>> indicator
    ArithmeticalMeanIndicator()
    >>> values = indicator(
    ...     prediction=torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]]),
    ...     target=torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]]),
    ... )
    >>> values
    tensor([[1.0000, 1.0000, 0.5000],
            [3.0000, 3.0000, 1.0000]])

    ```
    """

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return arithmetical_mean_indicator(prediction, target)


class ClassicalRelativeIndicator(BaseRelativeIndicator):
    r"""Implement the classical relative indicator function.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.modules.loss import ClassicalRelativeIndicator
    >>> prediction = torch.randn(3, 5, requires_grad=True)
    >>> target = torch.randn(3, 5)
    >>> indicator = ClassicalRelativeIndicator()
    >>> indicator
    ClassicalRelativeIndicator()
    >>> values = indicator(
    ...     prediction=torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]]),
    ...     target=torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]]),
    ... )
    >>> values
    tensor([[2., 1., 0.],
            [3., 5., 1.]])

    ```
    """

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return classical_relative_indicator(prediction, target)


class ReversedRelativeIndicator(BaseRelativeIndicator):
    r"""Implement the reversed relative indicator function.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.modules.loss import ReversedRelativeIndicator
    >>> prediction = torch.randn(3, 5, requires_grad=True)
    >>> target = torch.randn(3, 5)
    >>> indicator = ReversedRelativeIndicator()
    >>> indicator
    ReversedRelativeIndicator()
    >>> values = indicator(
    ...     prediction=torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]]),
    ...     target=torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]]),
    ... )
    >>> values
    tensor([[0., 1., 1.],
            [3., 1., 1.]])

    ```
    """

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return reversed_relative_indicator(prediction, target)
