r"""Contain relative loss functions."""

from __future__ import annotations

__all__ = [
    "RelativeIndicatorRegistry",
    "arithmetical_mean_indicator",
    "classical_relative_indicator",
    "relative_loss",
    "reversed_relative_indicator",
]


from collections.abc import Callable
from typing import ClassVar

import torch
from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from karbonn.functional.reduction import reduce_loss

IndicatorType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def relative_loss(
    loss: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    eps: float = 1e-8,
    indicator: IndicatorType | str = "classical_relative",
) -> torch.Tensor:
    r"""Compute the relative loss.

    The indicators are designed based on
    https://en.wikipedia.org/wiki/Relative_change#Indicators_of_relative_change.

    Args:
        loss: The loss values. The tensor must have the same shape as
            the target.
        prediction: The predictions.
        target: The target values.
        reduction: The reduction strategy. The valid values are
            ``'mean'``, ``'none'``,  and ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum will be divided by the number of elements in the
            input, ``'sum'``: the output will be summed.
        eps: An arbitrary small strictly positive number to avoid
            undefined results when the indicator is zero.
        indicator: The name of the indicator function to use or its
            implementation.

    Returns:
        The computed relative loss.

    Raises:
        RuntimeError: if the loss and target shapes do not match.
        ValueError: if the reduction is not valid.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.functional import relative_loss
    >>> prediction = torch.randn(3, 5, requires_grad=True)
    >>> target = torch.randn(3, 5)
    >>> loss = relative_loss(
    ...     loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
    ...     prediction=prediction,
    ...     target=target,
    ... )
    >>> loss
    tensor(..., grad_fn=<MeanBackward0>)

    ```
    """
    if loss.shape != target.shape:
        msg = f"loss {loss.shape} and target {target.shape} shapes do not match"
        raise RuntimeError(msg)
    if isinstance(indicator, str):
        indicator = RelativeIndicatorRegistry.find_indicator(indicator)
    loss = loss.div(indicator(prediction, target).clamp(min=eps))
    return reduce_loss(loss, reduction)


def arithmetical_mean_indicator(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r"""Return the arithmetical mean change.

    Args:
        prediction: The predictions.
        target: The target values.

    Returns:
        The indicator values.
    """
    return target.abs().add(prediction.abs()).mul(0.5)


def classical_relative_indicator(
    prediction: torch.Tensor,  # noqa: ARG001
    target: torch.Tensor,
) -> torch.Tensor:
    r"""Return the classical relative change.

    Args:
        prediction: The predictions.
        target: The target values.

    Returns:
        The indicator values.
    """
    return target.abs()


def reversed_relative_indicator(
    prediction: torch.Tensor,
    target: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    r"""Return the reversed relative change.

    Args:
        prediction: The predictions.
        target: The target values.

    Returns:
        The indicator values.
    """
    return prediction.abs()


class RelativeIndicatorRegistry:
    r"""Implement a registry of indicator functions."""

    registry: ClassVar[dict[str, IndicatorType]] = {
        "arithmetical_mean": arithmetical_mean_indicator,
        "classical_relative": classical_relative_indicator,
        "reversed_relative": reversed_relative_indicator,
    }

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping(dict(self.registry.items())))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping(dict(self.registry.items())))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    @classmethod
    def add_indicator(cls, name: str, indicator: IndicatorType, exist_ok: bool = False) -> None:
        r"""Add an indicator for a given name.

        Args:
            name: The name for the indicator to add.
            indicator: The indicator to add.
            exist_ok: If ``False``, ``ValueError`` is raised if the
                name already exists. This parameter should be set to
                ``True`` to overwrite the indicator for a name.

        Raises:
            RuntimeError: if an indicator is already registered
                for the name and ``exist_ok=False``.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.functional.loss.relative import RelativeIndicatorRegistry
        >>> def my_indicator(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ...     return prediction.abs() + target.abs()
        ...
        >>> RelativeIndicatorRegistry.add_indicator("other", my_indicator)  # doctest: +SKIP

        ```
        """
        if name in cls.registry and not exist_ok:
            msg = (
                f"An indicator ({cls.registry[name]}) is already registered for the name "
                f"'{name}'. Please use `exist_ok=True` if you want to overwrite the indicator "
                "for this name"
            )
            raise RuntimeError(msg)
        cls.registry[name] = indicator

    @classmethod
    def available_indicators(cls) -> tuple[str, ...]:
        """Get the available indicators.

        Returns:
            The available indicators.

        Example usage:

        ```pycon
        >>> from karbonn.functional.loss.relative import RelativeIndicatorRegistry
        >>> RelativeIndicatorRegistry.available_indicators()
        ('arithmetical_mean', 'classical_relative', 'reversed_relative')

        ```
        """
        return tuple(cls.registry.keys())

    @classmethod
    def find_indicator(cls, name: str) -> IndicatorType:
        r"""Find the indicator associated to a name.

        Args:
            name: The indicator name.

        Returns:
            The indicator.

        Example usage:

        ```pycon

        >>> from karbonn.functional.loss.relative import RelativeIndicatorRegistry
        >>> RelativeIndicatorRegistry.find_indicator("arithmetical_mean")
        <function arithmetical_mean_indicator at 0x...>
        >>> RelativeIndicatorRegistry.find_indicator("classical_relative")
        <function classical_relative_indicator at 0x...>

        ```
        """
        if (indicator := cls.registry.get(name)) is not None:
            return indicator
        msg = f"Incorrect name: {name}"
        raise RuntimeError(msg)

    @classmethod
    def has_indicator(cls, name: str) -> bool:
        r"""Indicate if an indicator is registered for the given name.

        Args:
            name: The name to check.

        Returns:
            ``True`` if a random indicator is registered,
                otherwise ``False``.

        Example usage:

        ```pycon

        >>> from karbonn.functional.loss.relative import RelativeIndicatorRegistry
        >>> RelativeIndicatorRegistry.has_indicator("arithmetical_mean")
        True
        >>> RelativeIndicatorRegistry.has_indicator("missing")
        False

        ```
        """
        return name in cls.registry
