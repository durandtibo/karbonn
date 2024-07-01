r"""Contain utility functions for distributed computing."""

from __future__ import annotations

__all__ = ["is_distributed"]

from torch.distributed import is_available, is_initialized


def is_distributed() -> bool:
    r"""Indicate if the current process is part of a distributed group.

    Returns:
        ``True`` if the current process is part of a distributed
            group, otherwise ``False``.

    Example usage:

    ```pycon

    >>> from karbonn.distributed import is_distributed
    >>> is_distributed()

    ```
    """
    return is_available() and is_initialized()
