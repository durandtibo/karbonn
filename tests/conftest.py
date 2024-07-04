from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import pytest

from karbonn.utils.imports import check_ignite, is_ignite_available

if TYPE_CHECKING:
    from collections.abc import Generator


if is_ignite_available():
    from ignite.distributed import Parallel


@pytest.fixture(scope="session")
def parallel_gloo_2() -> Generator[Parallel, None, None]:
    check_ignite()
    with Parallel(
        backend="gloo",
        nproc_per_node=2,
        nnodes=1,
        master_addr="localhost",
        master_port=29507,
        timeout=datetime.timedelta(seconds=60),
    ) as parallel:
        yield parallel


@pytest.fixture(scope="session")
def parallel_nccl_2() -> Generator[Parallel, None, None]:
    check_ignite()
    with Parallel(
        backend="nccl",
        nproc_per_node=2,
        nnodes=1,
        master_addr="localhost",
        master_port=29508,
        timeout=datetime.timedelta(seconds=60),
    ) as parallel:
        yield parallel
