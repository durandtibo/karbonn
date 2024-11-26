r"""Contain utility functions for distributed computing."""

from __future__ import annotations

__all__ = ["UnknownBackendError", "distributed_context", "is_distributed", "is_main_process"]

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

from torch.distributed import get_rank, is_available, is_initialized

from karbonn.utils.imports import is_ignite_available

if TYPE_CHECKING:
    from collections.abc import Generator

if is_ignite_available():  # pragma: no cover
    from ignite import distributed as idist

logger = logging.getLogger(__name__)


class UnknownBackendError(Exception):
    r"""Raised when you try to use an unknown backend."""


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


@contextmanager
def distributed_context(backend: str) -> Generator[None]:
    r"""Context manager to initialize the distributed context for a given
    backend.

    Args:
        backend: The distributed backend to use. You can find more
            information on the distributed backends at
            https://pytorch.org/docs/stable/distributed.html#backends

    Example usage

    ```pycon

    >>> import torch
    >>> from karbonn import distributed as dist
    >>> from ignite import distributed as idist
    >>> with dist.distributed_context(backend="gloo"):
    ...     idist.backend()
    ...     x = torch.ones(2, 3, device=idist.device())
    ...     idist.all_reduce(x, op="SUM")
    ...

    ```
    """
    if backend not in idist.available_backends():
        msg = f"Unknown backend '{backend}'. Available backends: {idist.available_backends()}"
        raise UnknownBackendError(msg)

    idist.initialize(backend, init_method="env://")

    try:
        # Distributed processes synchronization is needed here to
        # prevent a possible timeout after calling init_process_group.
        # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
        idist.barrier()
        yield
    finally:
        logger.info("Destroying the distributed process...")
        idist.finalize()
        logger.info("Distributed process destroyed")
