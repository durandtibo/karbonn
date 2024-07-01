r"""Contain utility functions for distributed computing."""

from __future__ import annotations

__all__ = ["is_distributed", "is_main_process"]

from torch.distributed import get_rank, is_available, is_initialized


def is_distributed() -> bool:
    r"""Indicate if the current process is part of a distributed group.

    Returns:
        ``True`` if the current process is part of a distributed
            group, otherwise ``False``.

    Example usage:

    ```pycon

    >>> from karbonn.distributed import is_distributed
    >>> is_distributed()
    False

    ```
    """
    return is_available() and is_initialized()


def is_main_process() -> bool:
    r"""Indicate if this process is the main process.

    By definition, the main process is the process with the global
    rank 0.

    Returns:
        ``True`` if it is the main process, otherwise ``False``.

    Example usage:

    ```pycon

    >>> from karbonn.distributed import is_main_process
    >>> is_main_process()
    True

    ```
    """
    if not is_distributed():
        return True
    return get_rank() == 0
