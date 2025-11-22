from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from karbonn.utils.tracker import EmptyTrackerError, ScalarTracker

###################################
#     Tests for ScalarTracker     #
###################################


def test_scalar_tracker_repr() -> None:
    assert (
        repr(ScalarTracker())
        == "ScalarTracker(count=0.0, total=0.0, min_value=inf, max_value=-inf, max_size=100)"
    )


@patch("karbonn.utils.tracker.ScalarTracker.std", lambda *args: 1.5)  # noqa: ARG005
def test_scalar_tracker_str() -> None:
    assert str(
        ScalarTracker(total=6.0, count=2, max_value=4.0, min_value=2.0, values=(4.0, 2.0))
    ) == (
        """ScalarTracker(
  (average): 3.0
  (count): 2.0
  (max): 4.0
  (median): 2.0
  (min): 2.0
  (std): 1.5
  (sum): 6.0
)"""
    )


def test_scalar_tracker_str_empty() -> None:
    assert str(ScalarTracker()) == (
        """ScalarTracker(
  (average): N/A (empty)
  (count): 0.0
  (max): N/A (empty)
  (median): N/A (empty)
  (min): N/A (empty)
  (std): N/A (empty)
  (sum): N/A (empty)
)"""
    )


def test_scalar_tracker_count() -> None:
    assert (
        ScalarTracker(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        ).count
        == 10
    )


def test_scalar_tracker_count_empty() -> None:
    assert ScalarTracker().count == 0


def test_scalar_tracker_total() -> None:
    assert (
        ScalarTracker(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        ).total
        == 122.0
    )


def test_scalar_tracker_total_empty() -> None:
    assert ScalarTracker().total == 0


def test_scalar_tracker_values() -> None:
    assert ScalarTracker(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).values == (1.0, 3.0, 5.0, 4.0, 2.0)


def test_scalar_tracker_values_empty() -> None:
    assert ScalarTracker().values == ()


def test_scalar_tracker_average() -> None:
    assert (
        ScalarTracker(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        ).average()
        == 12.2
    )


def test_scalar_tracker_average_empty() -> None:
    tracker = ScalarTracker()
    with pytest.raises(EmptyTrackerError, match=r"The tracker is empty"):
        tracker.average()


def test_scalar_tracker_clone() -> None:
    tracker = ScalarTracker(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    )
    tracker_cloned = tracker.clone()
    assert tracker is not tracker_cloned
    assert tracker_cloned.equal(
        ScalarTracker(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )
    tracker.update(5)
    assert tracker.equal(
        ScalarTracker(
            total=127.0,
            count=11,
            max_value=6.0,
            min_value=-2.0,
            values=(1.0, 3.0, 5.0, 4.0, 2.0, 5.0),
        )
    )
    assert tracker_cloned.equal(
        ScalarTracker(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_tracker_clone_empty() -> None:
    tracker = ScalarTracker()
    tracker_cloned = tracker.clone()
    assert tracker is not tracker_cloned
    assert tracker_cloned.equal(ScalarTracker())
    tracker.update(5)
    assert tracker.equal(
        ScalarTracker(total=5.0, count=1, max_value=5.0, min_value=5.0, values=(5.0,))
    )
    assert tracker_cloned.equal(ScalarTracker())


def test_scalar_tracker_equal_true() -> None:
    assert ScalarTracker(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).equal(
        ScalarTracker(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_tracker_equal_true_empty() -> None:
    assert ScalarTracker().equal(ScalarTracker())


def test_scalar_tracker_equal_false_different_count() -> None:
    assert not ScalarTracker(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).equal(
        ScalarTracker(
            total=122.0, count=9, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_tracker_equal_false_different_total() -> None:
    assert not ScalarTracker(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).equal(
        ScalarTracker(
            total=12.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_tracker_equal_false_different_max_value() -> None:
    assert not ScalarTracker(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).equal(
        ScalarTracker(
            total=122.0, count=10, max_value=16.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_tracker_equal_false_different_min_value() -> None:
    assert not ScalarTracker(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).equal(
        ScalarTracker(
            total=122.0, count=10, max_value=6.0, min_value=-12.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_tracker_equal_false_different_values() -> None:
    assert not ScalarTracker(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).equal(
        ScalarTracker(total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1, 2, 3, 4, 6))
    )


def test_scalar_tracker_equal_false_different_type() -> None:
    assert not ScalarTracker(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).equal(1)


def test_scalar_tracker_load_state_dict() -> None:
    tracker = ScalarTracker()
    tracker.load_state_dict(
        {
            "count": 10,
            "total": 122.0,
            "values": (1.0, 3.0, 5.0, 4.0, 2.0),
            "max_value": 6.0,
            "min_value": -2.0,
        }
    )
    assert tracker.equal(
        ScalarTracker(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_tracker_max() -> None:
    assert (
        ScalarTracker(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        ).max()
        == 6.0
    )


def test_scalar_tracker_max_empty() -> None:
    tracker = ScalarTracker()
    with pytest.raises(EmptyTrackerError, match=r"The tracker is empty"):
        tracker.max()


def test_scalar_tracker_median() -> None:
    assert (
        ScalarTracker(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        ).median()
        == 3.0
    )


def test_scalar_tracker_median_empty() -> None:
    tracker = ScalarTracker()
    with pytest.raises(EmptyTrackerError, match=r"The tracker is empty"):
        tracker.median()


def test_scalar_tracker_merge() -> None:
    tracker = ScalarTracker(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    )
    tracker_merged = tracker.merge(
        [
            ScalarTracker(total=12.0, count=5, max_value=8.0, min_value=-1.0, values=(1.0, 3.0)),
            ScalarTracker(),
            ScalarTracker(total=-5.0, count=2, max_value=2.0, min_value=-5.0, values=(-5.0, 2.0)),
        ]
    )
    assert tracker.equal(
        ScalarTracker(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )
    assert tracker_merged.equal(
        ScalarTracker(
            total=129.0, count=17, max_value=8.0, min_value=-5.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_tracker_merge_() -> None:
    tracker = ScalarTracker(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    )
    tracker.merge_(
        [
            ScalarTracker(total=12.0, count=5, max_value=8.0, min_value=-1.0, values=(1.0, 3.0)),
            ScalarTracker(),
            ScalarTracker(total=-5.0, count=2, max_value=2.0, min_value=-5.0, values=(-5.0, 2.0)),
        ]
    )
    assert tracker.equal(
        ScalarTracker(
            total=129.0, count=17, max_value=8.0, min_value=-5.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_tracker_min() -> None:
    assert (
        ScalarTracker(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        ).min()
        == -2.0
    )


def test_scalar_tracker_min_empty() -> None:
    tracker = ScalarTracker()
    with pytest.raises(EmptyTrackerError, match=r"The tracker is empty"):
        tracker.min()


def test_scalar_tracker_reset() -> None:
    tracker = ScalarTracker(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    )
    tracker.reset()
    assert tracker.equal(ScalarTracker())


def test_scalar_tracker_reset_empty() -> None:
    tracker = ScalarTracker()
    tracker.reset()
    assert tracker.equal(ScalarTracker())


def test_scalar_tracker_state_dict() -> None:
    assert ScalarTracker(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).state_dict() == {
        "count": 10,
        "total": 122.0,
        "values": (1.0, 3.0, 5.0, 4.0, 2.0),
        "max_value": 6.0,
        "min_value": -2.0,
    }


def test_scalar_tracker_state_dict_empty() -> None:
    assert ScalarTracker().state_dict() == {
        "count": 0,
        "total": 0.0,
        "values": (),
        "max_value": -float("inf"),
        "min_value": float("inf"),
    }


def test_scalar_tracker_std() -> None:
    assert (
        ScalarTracker(
            total=10.0, count=10, max_value=1.0, min_value=1.0, values=(1.0, 1.0, 1.0, 1.0, 1.0)
        ).std()
        == 0.0
    )


def test_scalar_tracker_std_empty() -> None:
    tracker = ScalarTracker()
    with pytest.raises(EmptyTrackerError, match=r"The tracker is empty"):
        tracker.std()


def test_scalar_tracker_sum() -> None:
    assert (
        ScalarTracker(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        ).sum()
        == 122.0
    )


def test_scalar_tracker_sum_empty() -> None:
    tracker = ScalarTracker()
    with pytest.raises(EmptyTrackerError, match=r"The tracker is empty"):
        tracker.sum()


def test_scalar_tracker_update_4() -> None:
    tracker = ScalarTracker()
    tracker.update(4)
    assert tracker.equal(
        ScalarTracker(count=1, total=4.0, min_value=4.0, max_value=4.0, values=(4.0,))
    )


def test_scalar_tracker_update_4_and_2() -> None:
    tracker = ScalarTracker()
    tracker.update(4)
    tracker.update(2)
    assert tracker.equal(
        ScalarTracker(count=2, total=6.0, min_value=2.0, max_value=4.0, values=(4.0, 2.0))
    )


def test_scalar_tracker_update_max_window_size_3() -> None:
    tracker = ScalarTracker(max_size=3)
    tracker.update(0)
    tracker.update(3)
    tracker.update(1)
    tracker.update(4)
    assert tracker.equal(
        ScalarTracker(count=4, total=8.0, min_value=0.0, max_value=4.0, values=(3.0, 1.0, 4.0))
    )


def test_scalar_tracker_update_nan() -> None:
    tracker = ScalarTracker()
    tracker.update(float("NaN"))
    assert math.isnan(tracker.total)
    assert tracker.count == 1


def test_scalar_tracker_update_inf() -> None:
    tracker = ScalarTracker()
    tracker.update(float("inf"))
    assert tracker.equal(
        ScalarTracker(
            count=1,
            total=float("inf"),
            min_value=float("inf"),
            max_value=float("inf"),
            values=(float("inf"),),
        )
    )


@pytest.mark.parametrize("values", [[3, 1, 2], (3, 1, 2), (3.0, 1.0, 2.0)])
def test_scalar_tracker_update_sequence(values: list[float] | tuple[float, ...]) -> None:
    tracker = ScalarTracker()
    tracker.update_sequence(values)
    assert tracker.equal(
        ScalarTracker(count=3, total=6.0, min_value=1.0, max_value=3.0, values=(3.0, 1.0, 2.0))
    )
