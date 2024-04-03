r"""Contain functionalities to analyze a ``torch.nn.Module``."""

from __future__ import annotations

__all__ = ["parse_batch_shape"]

from collections.abc import Mapping, Sequence
from typing import Any, overload

import torch

PARAMETER_NUM_UNITS = (" ", "K", "M", "B", "T")
UNKNOWN_SIZE = "?"
UNKNOWN_DTYPE = "?"


@overload
def parse_batch_shape(batch: torch.Tensor) -> torch.Size | None: ...  # pragma: no cover


@overload
def parse_batch_shape(
    batch: Sequence[torch.Tensor],
) -> tuple[torch.Size | None, ...]: ...  # pragma: no cover


@overload
def parse_batch_shape(
    batch: Mapping[str, torch.Tensor]
) -> dict[str, torch.Size | None]: ...  # pragma: no cover


def parse_batch_shape(
    batch: Any,
) -> tuple[torch.Size | None, ...] | dict[str, torch.Size | None] | torch.Size | None:
    r"""Parse the shapes of the tensors in the batch.

    The current implementation only parses the shapes of  tensor,
    list of tensors, and dictionary of tensors.

    Args:
        batch: The batch to parse.

    Returns:
        The shapes in the batch or ``None`` if it cannot parse the input.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.utils.summary.module import parse_batch_shape
    >>> parse_batch_shape(torch.ones(2, 3))
    torch.Size([2, 3])
    >>> parse_batch_shape([torch.ones(2, 3), torch.zeros(2)])
    (torch.Size([2, 3]), torch.Size([2]))
    >>> parse_batch_shape({"input1": torch.ones(2, 3), "input2": torch.zeros(2)})
    {'input1': torch.Size([2, 3]), 'input2': torch.Size([2])}

    ```
    """
    if torch.is_tensor(batch):
        return batch.shape
    if isinstance(batch, Sequence):
        return tuple(parse_batch_shape(item) for item in batch)
    if isinstance(batch, Mapping):
        return {key: parse_batch_shape(value) for key, value in batch.items()}
    return None
