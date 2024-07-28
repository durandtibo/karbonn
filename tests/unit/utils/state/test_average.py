from __future__ import annotations

import math
from unittest.mock import Mock, patch

import pytest

from karbonn.distributed.ddp import SUM
from karbonn.utils.state import AverageState, EmptyStateError

##################################
#     Tests for AverageState     #
##################################


def test_average_state_repr() -> None:
    assert repr(AverageState()) == "AverageState(count=0.0, total=0.0)"


def test_average_state_str() -> None:
    assert (
        str(AverageState(total=6, count=2))
        == """AverageState(
  (average): 3.0
  (count): 2.0
  (total): 6.0
)"""
    )


def test_average_state_str_empty() -> None:
    assert str(AverageState()) == (
        """AverageState(
  (average): N/A (empty)
  (count): 0.0
  (total): 0.0
)"""
    )


def test_average_state_all_reduce() -> None:
    state = AverageState(total=122.0, count=10)
    state_reduced = state.all_reduce()
    assert state.equal(AverageState(total=122.0, count=10))
    assert state_reduced.equal(AverageState(total=122.0, count=10))


def test_average_state_all_reduce_empty() -> None:
    assert AverageState().all_reduce().equal(AverageState())


def test_average_state_all_reduce_total_reduce() -> None:
    state = AverageState(total=122.0, count=10)
    reduce_mock = Mock(side_effect=lambda *args: args[0] + 1)
    with patch("karbonn.utils.state.average.sync_reduce", reduce_mock):
        state_reduced = state.all_reduce()
        assert state.equal(AverageState(total=122.0, count=10))
        assert state_reduced.equal(AverageState(total=123.0, count=11))
        assert reduce_mock.call_args_list == [((122.0, SUM), {}), ((10, SUM), {})]


def test_average_state_average() -> None:
    assert AverageState(total=6, count=2).average() == 3.0


def test_average_state_average_empty() -> None:
    state = AverageState()
    with pytest.raises(EmptyStateError, match="The state is empty"):
        state.average()


def test_average_state_clone() -> None:
    state = AverageState(total=122.0, count=10)
    state_cloned = state.clone()
    assert state is not state_cloned
    assert state_cloned.equal(AverageState(total=122.0, count=10))


def test_average_state_clone_empty() -> None:
    state = AverageState()
    state_cloned = state.clone()
    assert state is not state_cloned
    assert state_cloned.equal(AverageState())


def test_average_state_equal_true() -> None:
    assert AverageState(total=122.0, count=10).equal(AverageState(total=122.0, count=10))


def test_average_state_equal_true_empty() -> None:
    assert AverageState().equal(AverageState())


def test_average_state_equal_false_different_total() -> None:
    assert not AverageState(total=121.0, count=10).equal(AverageState(total=122.0, count=10))


def test_average_state_equal_false_different_count() -> None:
    assert not AverageState(total=122.0, count=10).equal(AverageState(total=122.0, count=9))


def test_average_state_equal_false_different_type() -> None:
    assert not AverageState(total=122.0, count=10).equal(1)


def test_average_state_merge() -> None:
    state = AverageState(total=122.0, count=10)
    state_merged = state.merge(
        [AverageState(total=1.0, count=4), AverageState(), AverageState(total=-2.0, count=2)]
    )
    assert state.equal(AverageState(total=122.0, count=10))
    assert state_merged.equal(AverageState(total=121.0, count=16))


def test_average_state_merge_() -> None:
    state = AverageState(total=122.0, count=10)
    state.merge_(
        [AverageState(total=1.0, count=4), AverageState(), AverageState(total=-2.0, count=2)]
    )
    assert state.equal(AverageState(total=121.0, count=16))


def test_average_state_load_state_dict() -> None:
    state = AverageState()
    state.load_state_dict({"count": 10, "total": 122.0})
    assert state.equal(AverageState(total=122.0, count=10))


def test_average_state_reset() -> None:
    state = AverageState(total=122.0, count=10)
    state.reset()
    assert state.equal(AverageState())


def test_average_state_reset_empty() -> None:
    state = AverageState()
    state.reset()
    assert state.equal(AverageState())


def test_average_state_state_dict() -> None:
    assert AverageState(total=19.0, count=4).state_dict() == {"count": 4, "total": 19}


def test_average_state_state_dict_empty() -> None:
    assert AverageState().state_dict() == {"count": 0, "total": 0}


def test_average_state_sum() -> None:
    assert AverageState(total=6, count=2).sum() == 6.0


def test_average_state_sum_empty() -> None:
    state = AverageState()
    with pytest.raises(EmptyStateError, match="The state is empty"):
        state.sum()


def test_average_state_update_4() -> None:
    state = AverageState()
    state.update(4)
    assert state.equal(AverageState(total=4.0, count=1))


def test_average_state_update_4_and_2() -> None:
    state = AverageState()
    state.update(4)
    state.update(2)
    assert state.equal(AverageState(total=6.0, count=2))


def test_average_state_update_with_num_examples() -> None:
    state = AverageState()
    state.update(4, num_examples=2)
    state.update(2)
    state.update(2)
    assert state.equal(AverageState(total=12.0, count=4))


def test_average_state_update_nan() -> None:
    state = AverageState()
    state.update(float("NaN"))
    assert math.isnan(state.total)
    assert state.count == 1


def test_average_state_update_inf() -> None:
    state = AverageState()
    state.update(float("inf"))
    assert state.equal(AverageState(total=float("inf"), count=1))
