from __future__ import annotations

from torch.distributed import Backend

from karbonn.distributed import (
    distributed_context,
    gloocontext,
    is_distributed,
    is_main_process,
    ncclcontext,
)
from karbonn.testing import (
    cuda_available,
    distributed_available,
    gloo_available,
    ignite_available,
    nccl_available,
)
from karbonn.utils.imports import is_ignite_available

if is_ignite_available():  # pragma: no cover
    from ignite import distributed as idist


####################################
#     Tests for is_distributed     #
####################################


def test_is_distributed() -> None:
    assert not is_distributed()


#####################################
#     Tests for is_main_process     #
#####################################


def test_is_main_process() -> None:
    assert is_main_process()


########################################
#    Tests for distributed_context     #
########################################


@ignite_available
@distributed_available
@gloo_available
def test_distributed_context_gloo() -> None:
    with distributed_context(Backend.GLOO):
        assert idist.backend() == Backend.GLOO
    assert idist.backend() is None


@ignite_available
@distributed_available
@cuda_available
@nccl_available
def test_distributed_context_nccl() -> None:
    with distributed_context(Backend.NCCL):
        assert idist.backend() == Backend.NCCL
    assert idist.backend() is None


#################################
#     Tests for gloocontext     #
#################################


@ignite_available
@distributed_available
@gloo_available
def test_gloocontext() -> None:
    with gloocontext():
        assert idist.backend() == Backend.GLOO
    assert idist.backend() is None


#################################
#     Tests for ncclcontext     #
#################################


@ignite_available
@distributed_available
@cuda_available
@nccl_available
def test_ncclcontext() -> None:
    with ncclcontext():
        assert idist.backend() == Backend.NCCL
    assert idist.backend() is None
