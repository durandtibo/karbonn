r"""Define some PyTest fixtures."""

from __future__ import annotations

__all__ = [
    "cuda_available",
    "distributed_available",
    "objectory_available",
    "two_gpus_available",
]

import pytest
import torch

from karbonn.utils.imports import is_objectory_available

cuda_available = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires a device with CUDA"
)
two_gpus_available = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs"
)
distributed_available = pytest.mark.skipif(
    not torch.distributed.is_available(), reason="Requires PyTorch distributed"
)

objectory_available = pytest.mark.skipif(not is_objectory_available(), reason="Require objectory")
