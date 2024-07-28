from __future__ import annotations

import math
from unittest.mock import Mock, patch

import pytest

from karbonn.distributed.ddp import SUM
from karbonn.utils.tracker import Average, EmptyTrackerError

#############################
#     Tests for Average     #
#############################


def test_average_repr() -> None:
    assert repr(Average()) == "Average(count=0.0, total=0.0)"


def test_average_str() -> None:
    assert (
        str(Average(total=6, count=2))
        == """Average(
  (average): 3.0
  (count): 2.0
  (total): 6.0
)"""
    )


def test_average_str_empty() -> None:
    assert str(Average()) == (
        """Average(
  (average): N/A (empty)
  (count): 0.0
  (total): 0.0
)"""
    )


def test_average_all_reduce() -> None:
    tracker = Average(total=122.0, count=10)
    tracker_reduced = tracker.all_reduce()
    assert tracker.equal(Average(total=122.0, count=10))
    assert tracker_reduced.equal(Average(total=122.0, count=10))


def test_average_all_reduce_empty() -> None:
    assert Average().all_reduce().equal(Average())


def test_average_all_reduce_total_reduce() -> None:
    tracker = Average(total=122.0, count=10)
    reduce_mock = Mock(side_effect=lambda *args: args[0] + 1)
    with patch("karbonn.utils.tracker.average.sync_reduce", reduce_mock):
        tracker_reduced = tracker.all_reduce()
        assert tracker.equal(Average(total=122.0, count=10))
        assert tracker_reduced.equal(Average(total=123.0, count=11))
        assert reduce_mock.call_args_list == [((122.0, SUM), {}), ((10, SUM), {})]


def test_average_average() -> None:
    assert Average(total=6, count=2).average() == 3.0


def test_average_average_empty() -> None:
    tracker = Average()
    with pytest.raises(EmptyTrackerError, match="The tracker is empty"):
        tracker.average()


def test_average_clone() -> None:
    tracker = Average(total=122.0, count=10)
    tracker_cloned = tracker.clone()
    assert tracker is not tracker_cloned
    assert tracker_cloned.equal(Average(total=122.0, count=10))


def test_average_clone_empty() -> None:
    tracker = Average()
    tracker_cloned = tracker.clone()
    assert tracker is not tracker_cloned
    assert tracker_cloned.equal(Average())


def test_average_equal_true() -> None:
    assert Average(total=122.0, count=10).equal(Average(total=122.0, count=10))


def test_average_equal_true_empty() -> None:
    assert Average().equal(Average())


def test_average_equal_false_different_total() -> None:
    assert not Average(total=121.0, count=10).equal(Average(total=122.0, count=10))


def test_average_equal_false_different_count() -> None:
    assert not Average(total=122.0, count=10).equal(Average(total=122.0, count=9))


def test_average_equal_false_different_type() -> None:
    assert not Average(total=122.0, count=10).equal(1)


def test_average_merge() -> None:
    tracker = Average(total=122.0, count=10)
    tracker_merged = tracker.merge(
        [Average(total=1.0, count=4), Average(), Average(total=-2.0, count=2)]
    )
    assert tracker.equal(Average(total=122.0, count=10))
    assert tracker_merged.equal(Average(total=121.0, count=16))


def test_average_merge_() -> None:
    tracker = Average(total=122.0, count=10)
    tracker.merge_([Average(total=1.0, count=4), Average(), Average(total=-2.0, count=2)])
    assert tracker.equal(Average(total=121.0, count=16))


def test_average_load_tracker_dict() -> None:
    tracker = Average()
    tracker.load_tracker_dict({"count": 10, "total": 122.0})
    assert tracker.equal(Average(total=122.0, count=10))


def test_average_reset() -> None:
    tracker = Average(total=122.0, count=10)
    tracker.reset()
    assert tracker.equal(Average())


def test_average_reset_empty() -> None:
    tracker = Average()
    tracker.reset()
    assert tracker.equal(Average())


def test_average_tracker_dict() -> None:
    assert Average(total=19.0, count=4).tracker_dict() == {"count": 4, "total": 19}


def test_average_tracker_dict_empty() -> None:
    assert Average().tracker_dict() == {"count": 0, "total": 0}


def test_average_sum() -> None:
    assert Average(total=6, count=2).sum() == 6.0


def test_average_sum_empty() -> None:
    tracker = Average()
    with pytest.raises(EmptyTrackerError, match="The tracker is empty"):
        tracker.sum()


def test_average_update_4() -> None:
    tracker = Average()
    tracker.update(4)
    assert tracker.equal(Average(total=4.0, count=1))


def test_average_update_4_and_2() -> None:
    tracker = Average()
    tracker.update(4)
    tracker.update(2)
    assert tracker.equal(Average(total=6.0, count=2))


def test_average_update_with_num_examples() -> None:
    tracker = Average()
    tracker.update(4, num_examples=2)
    tracker.update(2)
    tracker.update(2)
    assert tracker.equal(Average(total=12.0, count=4))


def test_average_update_nan() -> None:
    tracker = Average()
    tracker.update(float("NaN"))
    assert math.isnan(tracker.total)
    assert tracker.count == 1


def test_average_update_inf() -> None:
    tracker = Average()
    tracker.update(float("inf"))
    assert tracker.equal(Average(total=float("inf"), count=1))
