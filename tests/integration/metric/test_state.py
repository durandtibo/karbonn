from __future__ import annotations

from typing import Callable
from unittest.mock import Mock

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal

from karbonn.metric.state import (
    AccuracyState,
    ErrorState,
    ExtendedAccuracyState,
    ExtendedErrorState,
    MeanErrorState,
    NormalizedMeanSquaredErrorState,
    RootMeanErrorState,
)
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


###################################
#     Tests for AccuracyState     #
###################################


def check_accuracy_state(local_rank: int) -> None:
    r"""Check ``AccuracyState``.

    Args:
        local_rank: The local rank.
    """
    assert idist.get_world_size() == 2  # This test is valid only for 2 processes.

    state = AccuracyState()
    state.update(
        torch.tensor([0, 1, 1, 0, 0, 0]) if local_rank == 0 else torch.tensor([0, 0, 1, 0])
    )
    assert objects_are_equal(state.value(), {"accuracy": 0.3, "num_predictions": 10})


###########################################
#     Tests for ExtendedAccuracyState     #
###########################################


def check_extended_accuracy_state(local_rank: int) -> None:
    r"""Check ``ExtendedAccuracyState``.

    Args:
        local_rank: The local rank.
    """
    assert idist.get_world_size() == 2  # This test is valid only for 2 processes.

    state = ExtendedAccuracyState()
    state.update(
        torch.tensor([0, 1, 1, 0, 0, 0]) if local_rank == 0 else torch.tensor([0, 0, 1, 0])
    )
    assert objects_are_equal(
        state.value(),
        {
            "accuracy": 0.3,
            "error": 0.7,
            "num_correct_predictions": 3,
            "num_incorrect_predictions": 7,
            "num_predictions": 10,
        },
    )


################################
#     Tests for ErrorState     #
################################


def check_error_state(local_rank: int) -> None:
    r"""Check ``ErrorState``.

    Args:
        local_rank: The local rank.
    """
    assert idist.get_world_size() == 2  # This test is valid only for 2 processes.

    state = ErrorState()
    state.update(
        torch.tensor([0.0, 2.0, 1.0, 0.0, 2.0, 0.0])
        if local_rank == 0
        else torch.tensor([0.0, 3.0, 2.0, 0.0])
    )
    assert objects_are_equal(
        state.value(),
        {
            "mean": 1.0,
            "min": 0.0,
            "max": 3.0,
            "sum": 10.0,
            "num_predictions": 10,
        },
    )


########################################
#     Tests for ExtendedErrorState     #
########################################


def check_extended_error_state(local_rank: int) -> None:
    r"""Check ``ExtendedErrorState``.

    Args:
        local_rank: The local rank.
    """
    assert idist.get_world_size() == 2  # This test is valid only for 2 processes.

    state = ExtendedErrorState()
    state.update(
        torch.tensor([0.0, 2.0, 1.0, 0.0, 2.0, 0.0])
        if local_rank == 0
        else torch.tensor([0.0, 5.0, 0.0, 0.0])
    )
    assert objects_are_allclose(
        state.value(),
        {
            "mean": 1.0,
            "median": 0.0,
            "min": 0.0,
            "max": 5.0,
            "sum": 10.0,
            "std": 1.632993221282959,
            "num_predictions": 10,
        },
    )


####################################
#     Tests for MeanErrorState     #
####################################


def check_mean_error_state(local_rank: int) -> None:
    r"""Check ``MeanErrorState``.

    Args:
        local_rank: The local rank.
    """
    assert idist.get_world_size() == 2  # This test is valid only for 2 processes.

    state = MeanErrorState()
    state.update(
        torch.tensor([0.0, 2.0, 1.0, 0.0, 2.0, 0.0])
        if local_rank == 0
        else torch.tensor([0.0, 5.0, 0.0, 0.0])
    )
    assert objects_are_equal(state.value(), {"mean": 1.0, "num_predictions": 10})


########################################
#     Tests for RootMeanErrorState     #
########################################


def check_root_mean_error_state(local_rank: int) -> None:
    r"""Check ``RootMeanErrorState``.

    Args:
        local_rank: The local rank.
    """
    assert idist.get_world_size() == 2  # This test is valid only for 2 processes.

    state = RootMeanErrorState()
    state.update(
        torch.tensor([0.0, 40.0, 60.0, 0.0, 20.0, 0.0])
        if local_rank == 0
        else torch.tensor([0.0, 40.0, 0.0, 0.0])
    )
    assert objects_are_equal(state.value(), {"mean": 4.0, "num_predictions": 10})


#####################################################
#     Tests for NormalizedMeanSquaredErrorState     #
#####################################################


def check_normalized_mean_squared_error_state(local_rank: int) -> None:
    r"""Check ``NormalizedMeanSquaredErrorState``.

    Args:
        local_rank: The local rank.
    """
    assert idist.get_world_size() == 2  # This test is valid only for 2 processes.

    state = NormalizedMeanSquaredErrorState()
    if local_rank == 0:
        state.update(
            torch.tensor([0.0, 2.0, 2.0, 0.0, 2.0, 0.0]),
            torch.full(size=(4,), fill_value=2.0),
        )
    else:
        state.update(
            torch.tensor([0.0, 2.0, 4.0, 0.0]),
            torch.full(size=(4,), fill_value=2.0),
        )
    assert objects_are_equal(state.value(), {"mean": 1.0, "num_predictions": 10})


CHECKS = [
    check_accuracy_state,
    check_extended_accuracy_state,
    check_error_state,
    check_extended_error_state,
    check_mean_error_state,
    check_root_mean_error_state,
    check_normalized_mean_squared_error_state,
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
