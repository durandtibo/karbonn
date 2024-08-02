from __future__ import annotations

from typing import Callable
from unittest.mock import Mock

import pytest
import torch

from karbonn.testing import (
    distributed_available,
    gloo_available,
    ignite_available,
    nccl_available,
    two_gpus_available,
)
from karbonn.utils.imports import is_ignite_available
from karbonn.utils.tracker import (
    Average,
    ExtremaTensorTracker,
    MeanTensorTracker,
    ScalableTensorTracker,
    TensorTracker,
)

if is_ignite_available():
    from ignite import distributed as idist
else:  # pragma: no cover
    idist = Mock()


#############################
#     Tests for Average     #
#############################


def check_average(local_rank: int) -> None:
    r"""Check ``Average``.

    Args:
        local_rank: The local rank.
    """
    assert idist.get_world_size() == 2  # This test is valid only for 2 processes.
    idist.device()

    tracker = Average(count=8, total=20.0) if local_rank == 0 else Average(count=2, total=12.0)
    assert tracker.all_reduce().equal(Average(count=10, total=32.0))

    tracker = Average(count=8, total=20.0) if local_rank == 0 else Average()
    assert tracker.all_reduce().equal(Average(count=8, total=20.0))


###################################
#     Tests for ScalarTracker     #
###################################


#######################################
#     Tests for MeanTensorTracker     #
#######################################


def check_mean_tensor_tracker(local_rank: int) -> None:
    r"""Check ``MeanTensorTracker``.

    Args:
        local_rank: The local rank.
    """
    assert idist.get_world_size() == 2  # This test is valid only for 2 processes.
    idist.device()

    tracker = (
        MeanTensorTracker(count=8, total=20.0)
        if local_rank == 0
        else MeanTensorTracker(count=2, total=12.0)
    )
    assert tracker.all_reduce().equal(MeanTensorTracker(count=10, total=32.0))

    tracker = MeanTensorTracker(count=8, total=20.0) if local_rank == 0 else MeanTensorTracker()
    assert tracker.all_reduce().equal(MeanTensorTracker(count=8, total=20.0))


##########################################
#     Tests for ExtremaTensorTracker     #
##########################################


def check_extrema_tensor_tracker(local_rank: int) -> None:
    r"""Check ``ExtremaTensorTracker``.

    Args:
        local_rank: The local rank.
    """
    assert idist.get_world_size() == 2  # This test is valid only for 2 processes.
    idist.device()

    tracker = (
        ExtremaTensorTracker(count=8, min_value=-2.0, max_value=5.0)
        if local_rank == 0
        else ExtremaTensorTracker(count=2, min_value=-3.0, max_value=2.0)
    )
    assert tracker.all_reduce().equal(ExtremaTensorTracker(count=10, min_value=-3.0, max_value=5.0))

    tracker = (
        ExtremaTensorTracker(count=8, min_value=-2.0, max_value=5.0)
        if local_rank == 0
        else ExtremaTensorTracker()
    )
    assert tracker.all_reduce().equal(ExtremaTensorTracker(count=8, min_value=-2.0, max_value=5.0))


###################################
#     Tests for TensorTracker     #
###################################


def check_tensor_tracker(local_rank: int) -> None:
    r"""Check ``TensorTracker``.

    Args:
        local_rank: The local rank.
    """
    assert idist.get_world_size() == 2  # This test is valid only for 2 processes.
    idist.device()

    tracker = (
        TensorTracker(torch.arange(6, dtype=torch.float))
        if local_rank == 0
        else TensorTracker(torch.tensor([4.0, 1.0]))
    )
    assert tracker.all_reduce().equal(
        TensorTracker(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 1.0], dtype=torch.float))
    )

    tracker = (
        TensorTracker(torch.arange(6, dtype=torch.float)) if local_rank == 0 else TensorTracker()
    )
    assert tracker.all_reduce().equal(TensorTracker(torch.arange(6, dtype=torch.float)))


###########################################
#     Tests for ScalableTensorTracker     #
###########################################


def check_scalable_tensor_tracker(local_rank: int) -> None:
    r"""Check ``ScalableTensorTracker``.

    Args:
        local_rank: The local rank.
    """
    assert idist.get_world_size() == 2  # This test is valid only for 2 processes.
    idist.device()

    tracker = (
        ScalableTensorTracker(count=8, total=20.0, min_value=0.0, max_value=5.0)
        if local_rank == 0
        else ScalableTensorTracker(count=2, total=12.0, min_value=-2.0, max_value=3.0)
    )
    assert tracker.all_reduce().equal(
        ScalableTensorTracker(count=10, total=32.0, min_value=-2.0, max_value=5.0)
    )

    tracker = (
        ScalableTensorTracker(count=8, total=20.0, min_value=0.0, max_value=5.0)
        if local_rank == 0
        else ScalableTensorTracker()
    )
    assert tracker.all_reduce().equal(
        ScalableTensorTracker(count=8, total=20.0, min_value=0.0, max_value=5.0)
    )


###########################################
#     Tests for BinaryConfusionMatrix     #
###########################################


###############################################
#     Tests for MulticlassConfusionMatrix     #
###############################################


CHECKS = [
    check_average,
    check_mean_tensor_tracker,
    check_extrema_tensor_tracker,
    check_tensor_tracker,
    check_scalable_tensor_tracker,
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
