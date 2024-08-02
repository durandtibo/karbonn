from __future__ import annotations

import pytest

from karbonn.utils.tracker import (
    EmptyTrackerError,
    ExponentialMovingAverage,
    MovingAverage,
)

###################################
#     Tests for MovingAverage     #
###################################


def test_moving_average_repr() -> None:
    assert repr(MovingAverage()).startswith("MovingAverage(")


def test_moving_average_str() -> None:
    assert str(MovingAverage()).startswith("MovingAverage(")


def test_moving_average_values() -> None:
    assert MovingAverage(values=(4, 2, 1)).values == (4, 2, 1)


def test_moving_average_values_empty() -> None:
    assert MovingAverage().values == ()


def test_moving_average_window_size() -> None:
    assert MovingAverage(window_size=5).window_size == 5


def test_moving_average_window_size_default() -> None:
    assert MovingAverage().window_size == 20


def test_moving_average_clone() -> None:
    tracker = MovingAverage(values=(4, 2, 1), window_size=5)
    tracker_cloned = tracker.clone()
    assert tracker is not tracker_cloned
    assert tracker_cloned.equal(MovingAverage(values=(4, 2, 1), window_size=5))


def test_moving_average_clone_empty() -> None:
    tracker = MovingAverage()
    tracker_cloned = tracker.clone()
    assert tracker is not tracker_cloned
    assert tracker_cloned.equal(MovingAverage())


def test_moving_average_equal_true() -> None:
    assert MovingAverage(values=(4, 2, 1), window_size=5).equal(
        MovingAverage(values=(4, 2, 1), window_size=5)
    )


def test_moving_average_equal_true_empty() -> None:
    assert MovingAverage().equal(MovingAverage())


def test_moving_average_equal_false_different_values() -> None:
    assert not MovingAverage(values=(4, 2, 1), window_size=5).equal(
        MovingAverage(values=(4, 2, 2), window_size=5)
    )


def test_moving_average_equal_false_different_window_size() -> None:
    assert not MovingAverage(values=(4, 2, 1), window_size=5).equal(
        MovingAverage(values=(4, 2, 1), window_size=10)
    )


def test_moving_average_equal_false_different_type() -> None:
    assert not MovingAverage(values=(4, 2, 1), window_size=5).equal(1)


def test_moving_average_load_state_dict() -> None:
    tracker = MovingAverage()
    tracker.load_state_dict({"values": (4, 2, 1), "window_size": 5})
    assert tracker.equal(MovingAverage(values=(4, 2, 1), window_size=5))


def test_moving_average_reset() -> None:
    tracker = MovingAverage(values=(4, 2, 1), window_size=5)
    tracker.reset()
    assert tracker.equal(MovingAverage(window_size=5))


def test_moving_average_reset_empty() -> None:
    tracker = MovingAverage()
    tracker.reset()
    assert tracker.equal(MovingAverage())


def test_moving_average_smoothed_average() -> None:
    assert MovingAverage(values=(4, 2)).smoothed_average() == 3.0


def test_moving_average_smoothed_average_empty() -> None:
    tracker = MovingAverage()
    with pytest.raises(EmptyTrackerError, match="The tracker is empty"):
        tracker.smoothed_average()


def test_moving_average_state_dict() -> None:
    assert MovingAverage(values=(4, 2, 1), window_size=5).state_dict() == {
        "values": (4, 2, 1),
        "window_size": 5,
    }


def test_moving_average_state_dict_empty() -> None:
    assert MovingAverage().state_dict() == {"values": (), "window_size": 20}


def test_moving_average_update() -> None:
    tracker = MovingAverage()
    tracker.update(4)
    tracker.update(2)
    tracker.equal(MovingAverage(values=(4, 2)))


##############################################
#     Tests for ExponentialMovingAverage     #
##############################################


def test_exponential_moving_average_repr() -> None:
    assert repr(ExponentialMovingAverage()).startswith("ExponentialMovingAverage(")


def test_exponential_moving_average_str() -> None:
    assert str(ExponentialMovingAverage()).startswith("ExponentialMovingAverage(")


def test_exponential_moving_average_count() -> None:
    assert ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35).count == 10


def test_exponential_moving_average_count_empty() -> None:
    assert ExponentialMovingAverage().count == 0


def test_exponential_moving_average_clone() -> None:
    tracker = ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35)
    tracker_cloned = tracker.clone()
    assert tracker is not tracker_cloned
    assert tracker_cloned.equal(
        ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35)
    )


def test_exponential_moving_average_clone_empty() -> None:
    tracker = ExponentialMovingAverage()
    tracker_cloned = tracker.clone()
    assert tracker is not tracker_cloned
    assert tracker_cloned.equal(ExponentialMovingAverage())


def test_exponential_moving_average_equal_true() -> None:
    assert ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35).equal(
        ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35)
    )


def test_exponential_moving_average_equal_true_empty() -> None:
    assert ExponentialMovingAverage().equal(ExponentialMovingAverage())


def test_exponential_moving_average_equal_false_different_alpha() -> None:
    assert not ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35).equal(
        ExponentialMovingAverage(alpha=0.95, count=10, smoothed_average=1.35)
    )


def test_exponential_moving_average_equal_false_different_count() -> None:
    assert not ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35).equal(
        ExponentialMovingAverage(alpha=0.9, count=9, smoothed_average=1.35)
    )


def test_exponential_moving_average_equal_false_different_smoothed_average() -> None:
    assert not ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35).equal(
        ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=2.35)
    )


def test_exponential_moving_average_equal_false_different_type() -> None:
    assert not ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35).equal(1)


def test_exponential_moving_average_load_state_dict() -> None:
    tracker = ExponentialMovingAverage()
    tracker.load_state_dict({"alpha": 0.9, "count": 10, "smoothed_average": 1.35})
    assert tracker.equal(ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35))


def test_exponential_moving_average_reset() -> None:
    tracker = ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35)
    tracker.reset()
    assert tracker.equal(ExponentialMovingAverage(alpha=0.9))


def test_exponential_moving_average_reset_empty() -> None:
    tracker = ExponentialMovingAverage()
    tracker.reset()
    assert tracker.equal(ExponentialMovingAverage())


def test_exponential_moving_average_state_dict() -> None:
    assert ExponentialMovingAverage(alpha=0.9, count=10, smoothed_average=1.35).state_dict() == {
        "alpha": 0.9,
        "count": 10,
        "smoothed_average": 1.35,
    }


def test_exponential_moving_average_state_dict_empty() -> None:
    assert ExponentialMovingAverage().state_dict() == {
        "alpha": 0.98,
        "count": 0,
        "smoothed_average": 0.0,
    }


def test_exponential_moving_average_smoothed_average() -> None:
    assert ExponentialMovingAverage(smoothed_average=1.35, count=1).smoothed_average() == 1.35


def test_exponential_moving_average_smoothed_average_empty() -> None:
    tracker = ExponentialMovingAverage()
    with pytest.raises(EmptyTrackerError, match="The tracker is empty"):
        tracker.smoothed_average()


def test_exponential_moving_average_update() -> None:
    tracker = ExponentialMovingAverage()
    tracker.update(4)
    tracker.update(2)
    tracker.equal(ExponentialMovingAverage(alpha=0.98, count=2, smoothed_average=3.96))
