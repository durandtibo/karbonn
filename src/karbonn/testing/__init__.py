r"""Define some utility functions for testing."""

from __future__ import annotations

__all__ = [
    "cuda_available",
    "distributed_available",
    "objectory_available",
    "two_gpus_available",
]


from karbonn.testing.fixtures import (
    cuda_available,
    distributed_available,
    objectory_available,
    two_gpus_available,
)
