r"""Contain modules to encode scalar values with cosine and sine
representations."""

from __future__ import annotations

__all__ = ["AsinhCosSinScalarEncoder", "CosSinScalarEncoder"]

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from karbonn.utils.tensor import to_tensor

if TYPE_CHECKING:
    from collections.abc import Sequence


class CosSinScalarEncoder(nn.Module):
    r"""Implement a frequency/phase-shift scalar encoder where the
    periodic functions are cosine and sine.

    Args:
        frequency: The initial frequency values.
        phase_shift: The initial phase-shift values.
        learnable: If ``True`` the frequencies and phase-shift
            parameters are learnable, otherwise they are frozen.

    Shape:
        - Input: ``(*, 1)``, where ``*`` means any number of
            dimensions.
        - Output: ``(*, feature_size)``,  where ``*`` has the same
            shape as the input.

    Example usage:

    ```pycon
    >>> import torch
    >>> from karbonn import CosSinScalarEncoder
    >>> m = CosSinScalarEncoder(
    ...     frequency=torch.tensor([1.0, 2.0, 4.0]), phase_shift=torch.tensor([1.0, 3.0, -2.0])
    ... )
    >>> m
    CosSinScalarEncoder(dim=3, learnable=False)
    >>> out = m(torch.tensor([[0.0], [1.0], [2.0], [3.0]]))
    >>> out
    tensor([[ 0.8415, -0.9900, -0.4161],
            [ 0.9093,  0.2837, -0.4161],
            [ 0.1411,  0.7539,  0.9602],
            [-0.7568, -0.9111, -0.8391]])

    ```
    """

    def __init__(
        self,
        frequency: torch.Tensor | Sequence[float],
        phase_shift: torch.Tensor | Sequence[float],
        learnable: bool = False,
    ) -> None:
        super().__init__()
        frequency = to_tensor(frequency)
        phase_shift = to_tensor(phase_shift)
        if frequency.ndim != 1:
            msg = (
                f"Incorrect number of dimensions for frequency (shape={frequency.shape}). "
                f"frequency has to be a 1D tensor or equivalent."
            )
            raise ValueError(msg)
        if frequency.shape != phase_shift.shape:
            msg = (
                f"Incorrect shapes. The shape of frequency (shape={frequency.shape})"
                f"does not match with phase_shift (shape={phase_shift.shape})"
            )
            raise ValueError(msg)
        self.frequency = nn.Parameter(frequency, requires_grad=learnable)
        self.phase_shift = nn.Parameter(phase_shift, requires_grad=learnable)
        self._half_size = int(self.frequency.shape[0] // 2)

    @property
    def input_size(self) -> int:
        r"""Return the input feature size."""
        return 1

    @property
    def output_size(self) -> int:
        r"""Return the output feature size."""
        return self.frequency.shape[0]

    def extra_repr(self) -> str:
        return f"dim={self.frequency.shape[0]}, learnable={self.frequency.requires_grad}"

    def forward(self, scalar: torch.Tensor) -> torch.Tensor:
        features = self.frequency * scalar + self.phase_shift
        return torch.cat(
            (features[..., : self._half_size].sin(), features[..., self._half_size :].cos()),
            dim=-1,
        )

    @classmethod
    def create_rand_frequency(
        cls,
        num_frequencies: int,
        min_frequency: float,
        max_frequency: float,
        learnable: bool = True,
    ) -> CosSinScalarEncoder:
        r"""Create a ``CosSinScalarEncoder`` where the frequencies are
        uniformly initialized in a frequency range.

        Args:
            num_frequencies: The number of frequencies.
            min_frequency: The minimum frequency.
            max_frequency: The maximum frequency.
            learnable: If ``True`` the parameters are learnable,
                otherwise they are frozen.

        Returns:
            An instantiated ``CosSinScalarEncoder`` where the
                frequencies are uniformly initialized in a frequency
                range.

        Example usage:

        ```pycon
        >>> import torch
        >>> from karbonn import CosSinScalarEncoder
        >>> m = CosSinScalarEncoder.create_rand_frequency(num_frequencies=5, min_frequency=0.1, max_frequency=1.0)
        >>> m
        CosSinScalarEncoder(dim=10, learnable=True)

        ```
        """
        if num_frequencies < 1:
            msg = f"num_frequencies has to be greater or equal to 1 (received: {num_frequencies})"
            raise ValueError(msg)
        if min_frequency <= 0:
            msg = f"min_frequency has to be greater than 0 (received: {min_frequency})"
            raise ValueError(msg)
        if max_frequency < min_frequency:
            msg = (
                f"max_frequency has to be greater than min_frequency {min_frequency} "
                f"(received: {max_frequency})"
            )
            raise ValueError(msg)
        return cls(
            frequency=torch.rand(num_frequencies)
            .mul(max_frequency - min_frequency)
            .add(min_frequency)
            .repeat(2),
            phase_shift=torch.zeros(2 * num_frequencies),
            learnable=learnable,
        )

    @classmethod
    def create_rand_value_range(
        cls,
        num_frequencies: int,
        min_abs_value: float,
        max_abs_value: float,
        learnable: bool = True,
    ) -> CosSinScalarEncoder:
        r"""Create a ``CosSinScalarEncoder`` where the frequencies are
        uniformly initialized for a given value range.

        Args:
            num_frequencies: The number of frequencies.
            min_abs_value: The minimum absolute value to encode.
            max_abs_value: The maximum absolute value to encoder.
            learnable: If ``True`` the parameters are learnable,
                otherwise they are frozen.

        Returns:
            An instantiated ``CosSinScalarEncoder`` where the
                frequencies are uniformly initialized for a given
                value range.

        Example usage:

        ```pycon
        >>> import torch
        >>> from karbonn import CosSinScalarEncoder
        >>> m = CosSinScalarEncoder.create_rand_value_range(num_frequencies=5, min_abs_value=0.1, max_abs_value=1.0)
        >>> m
        CosSinScalarEncoder(dim=10, learnable=True)

        ```
        """
        return cls.create_rand_frequency(
            num_frequencies=num_frequencies,
            min_frequency=1 / max_abs_value,
            max_frequency=1 / min_abs_value,
            learnable=learnable,
        )

    @classmethod
    def create_linspace_frequency(
        cls,
        num_frequencies: int,
        min_frequency: float,
        max_frequency: float,
        learnable: bool = True,
    ) -> CosSinScalarEncoder:
        r"""Create a `CosSinScalarEncoder`` where the frequencies are
        evenly spaced in a frequency range.

        Args:
            num_frequencies: The number of frequencies.
            min_frequency: The minimum frequency.
            max_frequency: The maximum frequency.
            learnable: If ``True`` the parameters are learnable,
                otherwise they are frozen.

        Returns:
            An instantiated ``CosSinScalarEncoder`` where the
                frequencies are evenly spaced in a frequency range.

        Example usage:

        ```pycon
        >>> import torch
        >>> from karbonn import CosSinScalarEncoder
        >>> m = CosSinScalarEncoder.create_linspace_frequency(num_frequencies=5, min_frequency=0.1, max_frequency=1.0)
        >>> m
        CosSinScalarEncoder(dim=10, learnable=True)

        ```
        """
        if num_frequencies < 1:
            msg = f"num_frequencies has to be greater or equal to 1 (received: {num_frequencies})"
            raise ValueError(msg)
        if min_frequency <= 0:
            msg = f"min_frequency has to be greater than 0 (received: {min_frequency})"
            raise ValueError(msg)
        if max_frequency < min_frequency:
            msg = (
                f"max_frequency has to be greater than min_frequency {min_frequency} "
                f"(received: {max_frequency})"
            )
            raise ValueError(msg)
        return cls(
            frequency=torch.linspace(
                start=min_frequency, end=max_frequency, steps=num_frequencies
            ).repeat(2),
            phase_shift=torch.zeros(2 * num_frequencies),
            learnable=learnable,
        )

    @classmethod
    def create_linspace_value_range(
        cls,
        num_frequencies: int,
        min_abs_value: float,
        max_abs_value: float,
        learnable: bool = True,
    ) -> CosSinScalarEncoder:
        r"""Create a ``CosSinScalarEncoder`` where the frequencies are
        evenly spaced given a value range.

        Args:
            num_frequencies: The number of frequencies.
            min_abs_value: The minimum absolute value to encode.
            max_abs_value: The maximum absolute value to encoder.
            learnable: If ``True`` the parameters are learnable,
                otherwise they are frozen.

        Returns:
            An instantiated ``CosSinScalarEncoder`` where the
                frequencies are evenly spaced.

        Example usage:

        ```pycon
        >>> import torch
        >>> from karbonn import CosSinScalarEncoder
        >>> m = CosSinScalarEncoder.create_linspace_value_range(num_frequencies=5, min_abs_value=0.1, max_abs_value=1.0)
        >>> m
        CosSinScalarEncoder(dim=10, learnable=True)

        ```
        """
        return cls.create_linspace_frequency(
            num_frequencies=num_frequencies,
            min_frequency=1 / max_abs_value,
            max_frequency=1 / min_abs_value,
            learnable=learnable,
        )

    @classmethod
    def create_logspace_frequency(
        cls,
        num_frequencies: int,
        min_frequency: float,
        max_frequency: float,
        learnable: bool = True,
    ) -> CosSinScalarEncoder:
        r"""Create a ``CosSinScalarEncoder`` where the frequencies are
        evenly spaced in the log space in a frequency range.

        Args:
            num_frequencies: The number of frequencies.
            min_frequency: The minimum frequency.
            max_frequency: The maximum frequency.
            learnable: If ``True`` the parameters are learnable,
                otherwise they are frozen.

        Returns:
            An instantiated ``CosSinScalarEncoder`` where the
                frequencies are evenly spaced in the log space in a
                frequency range.

        Example usage:

        ```pycon
        >>> import torch
        >>> from karbonn import CosSinScalarEncoder
        >>> m = CosSinScalarEncoder.create_logspace_frequency(num_frequencies=5, min_frequency=0.1, max_frequency=1.0)
        >>> m
        CosSinScalarEncoder(dim=10, learnable=True)

        ```
        """
        if num_frequencies < 1:
            msg = f"num_frequencies has to be greater or equal to 1 (received: {num_frequencies})"
            raise ValueError(msg)
        if min_frequency <= 0:
            msg = f"min_frequency has to be greater than 0 (received: {min_frequency})"
            raise ValueError(msg)
        if max_frequency < min_frequency:
            msg = (
                f"max_frequency has to be greater than min_frequency {min_frequency} "
                f"(received: {max_frequency})"
            )
            raise ValueError(msg)
        return cls(
            frequency=torch.logspace(
                start=math.log10(min_frequency),
                end=math.log10(max_frequency),
                steps=num_frequencies,
            ).repeat(2),
            phase_shift=torch.zeros(2 * num_frequencies),
            learnable=learnable,
        )

    @classmethod
    def create_logspace_value_range(
        cls,
        num_frequencies: int,
        min_abs_value: float,
        max_abs_value: float,
        learnable: bool = True,
    ) -> CosSinScalarEncoder:
        r"""Create a ``CosSinScalarEncoder`` where the frequencies are
        evenly spaced in the log space given a value range.

        Args:
            num_frequencies: The number of frequencies.
            min_abs_value: The minimum absolute value to encode.
            max_abs_value: The maximum absolute value to encoder.
            learnable: If ``True`` the parameters are learnable,
                otherwise they are frozen.

        Returns:
            An instantiated ``CosSinScalarEncoder`` where the
                frequencies are evenly spaced in the log space.

        Example usage:

        ```pycon
        >>> import torch
        >>> from karbonn import CosSinScalarEncoder
        >>> m = CosSinScalarEncoder.create_logspace_value_range(num_frequencies=5, min_abs_value=0.1, max_abs_value=1.0)
        >>> m
        CosSinScalarEncoder(dim=10, learnable=True)

        ```
        """
        return cls.create_logspace_frequency(
            num_frequencies=num_frequencies,
            min_frequency=1 / max_abs_value,
            max_frequency=1 / min_abs_value,
            learnable=learnable,
        )


class AsinhCosSinScalarEncoder(CosSinScalarEncoder):
    r"""Extension of ``CosSinScalarEncoder`` with an additional feature
    built using the inverse hyperbolic sine (arcsinh).

    Example usage:

    ```pycon
    >>> import torch
    >>> from karbonn import AsinhCosSinScalarEncoder
    >>> m = AsinhCosSinScalarEncoder(
    ...     frequency=torch.tensor([1.0, 2.0, 4.0]), phase_shift=torch.tensor([1.0, 3.0, -2.0])
    ... )
    >>> m
    AsinhCosSinScalarEncoder(dim=3, learnable=False)
    >>> out = m(torch.tensor([[0.0], [1.0], [2.0], [3.0]]))
    >>> out
    tensor([[ 0.8415, -0.9900, -0.4161,  0.0000],
            [ 0.9093,  0.2837, -0.4161,  0.8814],
            [ 0.1411,  0.7539,  0.9602,  1.4436],
            [-0.7568, -0.9111, -0.8391,  1.8184]])

    ```
    """

    @property
    def output_size(self) -> int:
        r"""Return the output feature size."""
        return self.frequency.shape[0] + 1

    def forward(self, scalar: torch.Tensor) -> torch.Tensor:
        features = self.frequency * scalar + self.phase_shift
        return torch.cat(
            (
                features[..., : self._half_size].sin(),
                features[..., self._half_size :].cos(),
                scalar.asinh(),
            ),
            dim=-1,
        )
