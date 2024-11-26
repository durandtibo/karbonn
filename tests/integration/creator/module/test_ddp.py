from __future__ import annotations

from unittest.mock import patch

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from karbonn.creator.module.ddp import to_ddp
from karbonn.testing import (
    cuda_available,
    gloo_available,
    ignite_available,
    nccl_available,
)
from karbonn.utils import get_module_device

############################
#     Tests for to_ddp     #
############################


@ignite_available
@gloo_available
def test_to_ddp_already_ddp() -> None:
    with gloocontext():
        module = to_ddp(DistributedDataParallel(nn.Linear(4, 5)))
        assert isinstance(module, DistributedDataParallel)
        assert isinstance(module.module, nn.Linear)


@ignite_available
@gloo_available
def test_to_ddp_linear_gloo() -> None:
    with gloocontext():
        module = to_ddp(nn.Linear(4, 5))
        assert isinstance(module, DistributedDataParallel)
        assert isinstance(module.module, nn.Linear)


@ignite_available
@cuda_available
@nccl_available
@patch("karbonn.creator.module.ddp.idist.get_world_size", lambda: 2)
def test_to_ddp_linear_nccl() -> None:
    with ncclcontext():
        module = to_ddp(nn.Linear(4, 5).to(device=torch.device("cuda:0")))
        assert isinstance(module, DistributedDataParallel)
        assert isinstance(module.module, nn.Linear)
        assert get_module_device(module) == torch.device("cuda:0")
