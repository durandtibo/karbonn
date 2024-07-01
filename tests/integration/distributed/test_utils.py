from __future__ import annotations

from karbonn.distributed import is_distributed

####################################
#     Tests for is_distributed     #
####################################


def test_is_distributed_false() -> None:
    assert not is_distributed()
