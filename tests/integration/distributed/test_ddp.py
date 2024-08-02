from __future__ import annotations

from typing import Callable
from unittest.mock import Mock

import pytest
import torch
from coola import objects_are_equal

from karbonn.distributed import ddp
from karbonn.testing import (
    distributed_available,
    gloo_available,
    ignite_available,
    nccl_available,
    two_gpus_available,
)
from karbonn.utils.imports import is_ignite_available

if is_ignite_available():
    from ignite import distributed as idist
else:  # pragma: no cover
    idist = Mock()


#################################
#     Tests for sync_reduce     #
#################################


def check_sync_reduce_tensor_int(local_rank: int) -> None:
    r"""Check ``sync_reduce`` for an integer tensor input.

    Args:
        local_rank: The local rank.
    """
    assert idist.get_world_size() == 2  # This test is valid only for 2 processes.
    device = idist.device()

    x_tensor = (
        torch.tensor([0, 1], device=device)
        if local_rank == 0
        else torch.tensor([2, 2], device=device)
    )
    assert ddp.sync_reduce(x_tensor, op=ddp.MAX).equal(torch.tensor([2, 2], device=device))  # max
    assert ddp.sync_reduce(x_tensor, op=ddp.MIN).equal(torch.tensor([0, 1], device=device))  # min
    assert ddp.sync_reduce(x_tensor, op=ddp.PRODUCT).equal(
        torch.tensor([0, 2], device=device)
    )  # product
    assert ddp.sync_reduce(x_tensor, op=ddp.SUM).equal(torch.tensor([2, 3], device=device))  # sum

    if idist.backend() != "nccl":  # bitwise AND and OR are not supported by NCCL
        assert ddp.sync_reduce(x_tensor, op=ddp.BAND).equal(
            torch.tensor([0, 0], device=device)
        )  # bitwise AND
        assert ddp.sync_reduce(x_tensor, op=ddp.BOR).equal(
            torch.tensor([2, 3], device=device)
        )  # bitwise OR

    # Verify that the original value did not change
    if local_rank == 0:
        assert x_tensor.equal(torch.tensor([0, 1], device=device))
    else:
        assert x_tensor.equal(torch.tensor([2, 2], device=device))


def check_sync_reduce_tensor_float(local_rank: int) -> None:
    r"""Check ``sync_reduce`` for a float tensor input.

    Args:
        local_rank: The local rank.
    """
    assert idist.get_world_size() == 2  # This test is valid only for 2 processes.
    device = idist.device()

    x_tensor = (
        torch.tensor([0.0, 1.0], device=device)
        if local_rank == 0
        else torch.tensor([2.0, 2.0], device=device)
    )
    assert ddp.sync_reduce(x_tensor, op=ddp.MAX).equal(
        torch.tensor([2.0, 2.0], device=device)
    )  # max
    assert ddp.sync_reduce(x_tensor, op=ddp.MIN).equal(
        torch.tensor([0.0, 1.0], device=device)
    )  # min
    assert ddp.sync_reduce(x_tensor, op=ddp.PRODUCT).equal(
        torch.tensor([0.0, 2.0], device=device)
    )  # product
    assert ddp.sync_reduce(x_tensor, op=ddp.SUM).equal(
        torch.tensor([2.0, 3.0], device=device)
    )  # sum

    with pytest.raises(RuntimeError, match="Cannot use ReduceOp.BAND with non-integral dtype"):
        ddp.sync_reduce(x_tensor, op=ddp.BAND)  # bitwise AND is not valid for float number
    with pytest.raises(RuntimeError, match="Cannot use ReduceOp.BOR with non-integral dtype"):
        ddp.sync_reduce(x_tensor, op=ddp.BOR)  # bitwise OR is not valid for float number

    # Verify that the original value did not change
    if local_rank == 0:
        assert x_tensor.equal(torch.tensor([0.0, 1.0], device=device))
    else:
        assert x_tensor.equal(torch.tensor([2.0, 2.0], device=device))


def check_sync_reduce_int(local_rank: int) -> None:
    r"""Check ``sync_reduce`` for a python integer input.

    Args:
        local_rank: The local rank.
    """
    assert idist.get_world_size() == 2  # This test is valid only for 2 processes.

    x_int = 2 if local_rank == 0 else 5
    assert ddp.sync_reduce(x_int, op=ddp.MAX) == 5  # max
    assert ddp.sync_reduce(x_int, op=ddp.MIN) == 2  # min
    assert ddp.sync_reduce(x_int, op=ddp.PRODUCT) == 10  # product
    assert ddp.sync_reduce(x_int, op=ddp.SUM) == 7  # sum

    if idist.backend() != "nccl":  # bitwise AND and OR are not supported by NCCL
        assert ddp.sync_reduce(x_int, op=ddp.BAND) == 0  # bitwise AND
        assert ddp.sync_reduce(x_int, op=ddp.BOR) == 7  # bitwise OR

    # Verify that the original value did not change
    if local_rank == 0:
        assert x_int == 2
    else:
        assert x_int == 5


def check_sync_reduce_float(local_rank: int) -> None:
    r"""Check ``sync_reduce`` for a python integer float.

    It also checks that the operations are not done in-place i.e.
    the original value did not change.

    Args:
        local_rank: The local rank.
    """
    assert idist.get_world_size() == 2  # This test is valid only for 2 processes.

    x_float = 1.0 if local_rank == 0 else 3.5
    assert ddp.sync_reduce(x_float, op=ddp.MAX) == 3.5  # max
    assert ddp.sync_reduce(x_float, op=ddp.MIN) == 1.0  # min
    assert ddp.sync_reduce(x_float, op=ddp.PRODUCT) == 3.5  # product
    assert ddp.sync_reduce(x_float, op=ddp.SUM) == 4.5  # sum

    with pytest.raises(RuntimeError, match="Cannot use ReduceOp.BAND with non-integral dtype"):
        ddp.sync_reduce(x_float, op=ddp.BAND)  # bitwise AND is not valid for float number
    with pytest.raises(RuntimeError, match="Cannot use ReduceOp.BOR with non-integral dtype"):
        ddp.sync_reduce(x_float, op=ddp.BOR)  # bitwise OR is not valid for float number

    # Verify that the original value did not change
    if local_rank == 0:
        assert x_float == 1.0
    else:
        assert x_float == 3.5


##################################
#     Tests for sync_reduce_     #
##################################


def check_sync_reduce_inplace(local_rank: int) -> None:
    r"""Check ``sync_reduce_`` which is the in-place version of
    ``sync_reduce``.

    Args:
        local_rank: The local rank.
    """
    assert idist.get_world_size() == 2
    device = idist.device()

    x_tensor = (
        torch.tensor([0, 1], device=device)
        if local_rank == 0
        else torch.tensor([2, 2], device=device)
    )
    assert ddp.sync_reduce_(x_tensor, op=ddp.MAX).equal(torch.tensor([2, 2], device=device))  # max
    assert x_tensor.equal(torch.tensor([2, 2], device=device))

    x_tensor = (
        torch.tensor([0, 1], device=device)
        if local_rank == 0
        else torch.tensor([2, 2], device=device)
    )
    assert ddp.sync_reduce_(x_tensor, op=ddp.MIN).equal(torch.tensor([0, 1], device=device))  # min
    assert x_tensor.equal(torch.tensor([0, 1], device=device))

    x_tensor = (
        torch.tensor([0, 1], device=device)
        if local_rank == 0
        else torch.tensor([2, 2], device=device)
    )
    assert ddp.sync_reduce_(x_tensor, op=ddp.PRODUCT).equal(
        torch.tensor([0, 2], device=device)
    )  # product
    assert x_tensor.equal(torch.tensor([0, 2], device=device))

    x_tensor = (
        torch.tensor([0, 1], device=device)
        if local_rank == 0
        else torch.tensor([2, 2], device=device)
    )
    assert ddp.sync_reduce_(x_tensor, op=ddp.SUM).equal(torch.tensor([2, 3], device=device))  # sum
    assert x_tensor.equal(torch.tensor([2, 3], device=device))

    if idist.backend() != "nccl":  # bitwise AND and OR are not supported by NCCL
        x_tensor = (
            torch.tensor([0, 1], device=device)
            if local_rank == 0
            else torch.tensor([2, 2], device=device)
        )
        assert ddp.sync_reduce_(x_tensor, op=ddp.BAND).equal(
            torch.tensor([0, 0], device=device)
        )  # bitwise AND
        assert x_tensor.equal(torch.tensor([0, 0], device=device))

        x_tensor = (
            torch.tensor([0, 1], device=device)
            if local_rank == 0
            else torch.tensor([2, 2], device=device)
        )
        assert ddp.sync_reduce_(x_tensor, op=ddp.BOR).equal(
            torch.tensor([2, 3], device=device)
        )  # bitwise OR
        assert x_tensor.equal(torch.tensor([2, 3], device=device))

    with pytest.raises(TypeError, match="sync_reduce_ only supports Tensor"):
        ddp.sync_reduce_(1.0, op=ddp.SUM)  # Does not support float


################################################
#     Tests for all_gather_tensor_varshape     #
################################################


def check_all_gather_tensor_varshape(local_rank: int) -> None:
    assert idist.get_world_size() == 2
    device = idist.device()

    # 1-d tensor
    assert objects_are_equal(
        ddp.all_gather_tensor_varshape(
            torch.tensor([0.0, 1.0]) if local_rank == 0 else torch.tensor([2.0, 3.0, 4.0])
        ),
        [
            torch.tensor([0.0, 1.0], dtype=torch.float, device=device),
            torch.tensor([2.0, 3.0, 4.0], dtype=torch.float, device=device),
        ],
    )

    # 2-d tensor
    assert objects_are_equal(
        ddp.all_gather_tensor_varshape(
            torch.tensor([[0, 1, 2], [3, 4, 5]]) if local_rank == 0 else torch.tensor([[1], [0]])
        ),
        [
            torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long, device=device),
            torch.tensor([[1], [0]], dtype=torch.long, device=device),
        ],
    )

    # 3-d tensor
    assert objects_are_equal(
        ddp.all_gather_tensor_varshape(
            torch.ones(2, 3, 4) if local_rank == 0 else torch.ones(4, 3, 1)
        ),
        [torch.ones(2, 3, 4, device=device), torch.ones(4, 3, 1, device=device)],
    )


CHECKS = [
    check_sync_reduce_tensor_int,
    check_sync_reduce_tensor_float,
    check_sync_reduce_int,
    check_sync_reduce_float,
    check_sync_reduce_inplace,
    check_all_gather_tensor_varshape,
]


@pytest.mark.parametrize("func", CHECKS)
@distributed_available
@gloo_available
@ignite_available
def test_sync_reduce_gloo(parallel_gloo_2: idist.Parallel, func: Callable[[int], None]) -> None:
    parallel_gloo_2.run(func)


@pytest.mark.parametrize("func", CHECKS)
@two_gpus_available
@distributed_available
@nccl_available
@ignite_available
def test_sync_reduce_nccl(parallel_nccl_2: idist.Parallel, func: Callable[[int], None]) -> None:
    parallel_nccl_2.run(func)
