from __future__ import annotations

import math
from unittest.mock import Mock, patch

import pytest

from karbonn.distributed.ddp import SUM
from karbonn.utils.tracker import AverageTracker, EmptyTrackerError

####################################
#     Tests for AverageTracker     #
####################################


def test_average_tracker_repr() -> None:
    assert repr(AverageTracker()) == "AverageTracker(count=0.0, total=0.0)"


def test_average_tracker_str() -> None:
    assert (
        str(AverageTracker(total=6, count=2))
        == """AverageTracker(
  (average): 3.0
  (count): 2.0
  (total): 6.0
)"""
    )


def test_average_tracker_str_empty() -> None:
    assert str(AverageTracker()) == (
        """AverageTracker(
  (average): N/A (empty)
  (count): 0.0
  (total): 0.0
)"""
    )


def test_average_tracker_all_reduce() -> None:
    tracker = AverageTracker(total=122.0, count=10)
    tracker_reduced = tracker.all_reduce()
    assert tracker.equal(AverageTracker(total=122.0, count=10))
    assert tracker_reduced.equal(AverageTracker(total=122.0, count=10))


def test_average_tracker_all_reduce_empty() -> None:
    assert AverageTracker().all_reduce().equal(AverageTracker())


def test_average_tracker_all_reduce_total_reduce() -> None:
    tracker = AverageTracker(total=122.0, count=10)
    reduce_mock = Mock(side_effect=lambda *args: args[0] + 1)
    with patch("karbonn.utils.tracker.average.sync_reduce", reduce_mock):
        tracker_reduced = tracker.all_reduce()
        assert tracker.equal(AverageTracker(total=122.0, count=10))
        assert tracker_reduced.equal(AverageTracker(total=123.0, count=11))
        assert reduce_mock.call_args_list == [((122.0, SUM), {}), ((10, SUM), {})]


def test_average_tracker_average() -> None:
    assert AverageTracker(total=6, count=2).average() == 3.0


def test_average_tracker_average_empty() -> None:
    tracker = AverageTracker()
    with pytest.raises(EmptyTrackerError, match=r"The tracker is empty"):
        tracker.average()


def test_average_tracker_clone() -> None:
    tracker = AverageTracker(total=122.0, count=10)
    tracker_cloned = tracker.clone()
    assert tracker is not tracker_cloned
    assert tracker_cloned.equal(AverageTracker(total=122.0, count=10))


def test_average_tracker_clone_empty() -> None:
    tracker = AverageTracker()
    tracker_cloned = tracker.clone()
    assert tracker is not tracker_cloned
    assert tracker_cloned.equal(AverageTracker())


def test_average_tracker_equal_true() -> None:
    assert AverageTracker(total=122.0, count=10).equal(AverageTracker(total=122.0, count=10))


def test_average_tracker_equal_true_empty() -> None:
    assert AverageTracker().equal(AverageTracker())


def test_average_tracker_equal_false_different_total() -> None:
    assert not AverageTracker(total=121.0, count=10).equal(AverageTracker(total=122.0, count=10))


def test_average_tracker_equal_false_different_count() -> None:
    assert not AverageTracker(total=122.0, count=10).equal(AverageTracker(total=122.0, count=9))


def test_average_tracker_equal_false_different_type() -> None:
    assert not AverageTracker(total=122.0, count=10).equal(1)


def test_average_tracker_merge() -> None:
    tracker = AverageTracker(total=122.0, count=10)
    tracker_merged = tracker.merge(
        [AverageTracker(total=1.0, count=4), AverageTracker(), AverageTracker(total=-2.0, count=2)]
    )
    assert tracker.equal(AverageTracker(total=122.0, count=10))
    assert tracker_merged.equal(AverageTracker(total=121.0, count=16))


def test_average_tracker_merge_() -> None:
    tracker = AverageTracker(total=122.0, count=10)
    tracker.merge_(
        [AverageTracker(total=1.0, count=4), AverageTracker(), AverageTracker(total=-2.0, count=2)]
    )
    assert tracker.equal(AverageTracker(total=121.0, count=16))


def test_average_tracker_load_state_dict() -> None:
    tracker = AverageTracker()
    tracker.load_state_dict({"count": 10, "total": 122.0})
    assert tracker.equal(AverageTracker(total=122.0, count=10))


def test_average_tracker_reset() -> None:
    tracker = AverageTracker(total=122.0, count=10)
    tracker.reset()
    assert tracker.equal(AverageTracker())


def test_average_tracker_reset_empty() -> None:
    tracker = AverageTracker()
    tracker.reset()
    assert tracker.equal(AverageTracker())


def test_average_tracker_state_dict() -> None:
    assert AverageTracker(total=19.0, count=4).state_dict() == {"count": 4, "total": 19}


def test_average_tracker_state_dict_empty() -> None:
    assert AverageTracker().state_dict() == {"count": 0, "total": 0}


def test_average_tracker_sum() -> None:
    assert AverageTracker(total=6, count=2).sum() == 6.0


def test_average_tracker_sum_empty() -> None:
    tracker = AverageTracker()
    with pytest.raises(EmptyTrackerError, match=r"The tracker is empty"):
        tracker.sum()


def test_average_tracker_update_4() -> None:
    tracker = AverageTracker()
    tracker.update(4)
    assert tracker.equal(AverageTracker(total=4.0, count=1))


def test_average_tracker_update_4_and_2() -> None:
    tracker = AverageTracker()
    tracker.update(4)
    tracker.update(2)
    assert tracker.equal(AverageTracker(total=6.0, count=2))


def test_average_tracker_update_with_num_examples() -> None:
    tracker = AverageTracker()
    tracker.update(4, num_examples=2)
    tracker.update(2)
    tracker.update(2)
    assert tracker.equal(AverageTracker(total=12.0, count=4))


def test_average_tracker_update_nan() -> None:
    tracker = AverageTracker()
    tracker.update(float("NaN"))
    assert math.isnan(tracker.total)
    assert tracker.count == 1


def test_average_tracker_update_inf() -> None:
    tracker = AverageTracker()
    tracker.update(float("inf"))
    assert tracker.equal(AverageTracker(total=float("inf"), count=1))
