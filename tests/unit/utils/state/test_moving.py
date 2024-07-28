from __future__ import annotations

import pytest

from karbonn.utils.state import (
    EmptyStateError,
    ExponentialMovingAverageState,
    MovingAverageState,
)

########################################
#     Tests for MovingAverageState     #
########################################


def test_moving_average_repr() -> None:
    assert repr(MovingAverageState()).startswith("MovingAverageState(")


def test_moving_average_str() -> None:
    assert str(MovingAverageState()).startswith("MovingAverageState(")


def test_moving_average_values() -> None:
    assert MovingAverageState(values=(4, 2, 1)).values == (4, 2, 1)


def test_moving_average_values_empty() -> None:
    assert MovingAverageState().values == ()


def test_moving_average_window_size() -> None:
    assert MovingAverageState(window_size=5).window_size == 5


def test_moving_average_window_size_default() -> None:
    assert MovingAverageState().window_size == 20


def test_moving_average_clone() -> None:
    state = MovingAverageState(values=(4, 2, 1), window_size=5)
    state_cloned = state.clone()
    assert state is not state_cloned
    assert state_cloned.equal(MovingAverageState(values=(4, 2, 1), window_size=5))


def test_moving_average_clone_empty() -> None:
    state = MovingAverageState()
    state_cloned = state.clone()
    assert state is not state_cloned
    assert state_cloned.equal(MovingAverageState())


def test_moving_average_equal_true() -> None:
    assert MovingAverageState(values=(4, 2, 1), window_size=5).equal(
        MovingAverageState(values=(4, 2, 1), window_size=5)
    )


def test_moving_average_equal_true_empty() -> None:
    assert MovingAverageState().equal(MovingAverageState())


def test_moving_average_equal_false_different_values() -> None:
    assert not MovingAverageState(values=(4, 2, 1), window_size=5).equal(
        MovingAverageState(values=(4, 2, 2), window_size=5)
    )


def test_moving_average_equal_false_different_window_size() -> None:
    assert not MovingAverageState(values=(4, 2, 1), window_size=5).equal(
        MovingAverageState(values=(4, 2, 1), window_size=10)
    )


def test_moving_average_equal_false_different_type() -> None:
    assert not MovingAverageState(values=(4, 2, 1), window_size=5).equal(1)


def test_moving_average_load_state_dict() -> None:
    state = MovingAverageState()
    state.load_state_dict({"values": (4, 2, 1), "window_size": 5})
    assert state.equal(MovingAverageState(values=(4, 2, 1), window_size=5))


def test_moving_average_reset() -> None:
    state = MovingAverageState(values=(4, 2, 1), window_size=5)
    state.reset()
    assert state.equal(MovingAverageState(window_size=5))


def test_moving_average_reset_empty() -> None:
    state = MovingAverageState()
    state.reset()
    assert state.equal(MovingAverageState())


def test_moving_average_smoothed_average() -> None:
    assert MovingAverageState(values=(4, 2)).smoothed_average() == 3.0


def test_moving_average_smoothed_average_empty() -> None:
    state = MovingAverageState()
    with pytest.raises(EmptyStateError, match="The state is empty"):
        state.smoothed_average()


def test_moving_average_state_dict() -> None:
    assert MovingAverageState(values=(4, 2, 1), window_size=5).state_dict() == {
        "values": (4, 2, 1),
        "window_size": 5,
    }


def test_moving_average_state_dict_empty() -> None:
    assert MovingAverageState().state_dict() == {"values": (), "window_size": 20}


def test_moving_average_update() -> None:
    state = MovingAverageState()
    state.update(4)
    state.update(2)
    state.equal(MovingAverageState(values=(4, 2)))


###################################################
#     Tests for ExponentialMovingAverageState     #
###################################################


def test_exponential_moving_average_repr() -> None:
    assert repr(ExponentialMovingAverageState()).startswith("ExponentialMovingAverageState(")


def test_exponential_moving_average_str() -> None:
    assert str(ExponentialMovingAverageState()).startswith("ExponentialMovingAverageState(")


def test_exponential_moving_average_count() -> None:
    assert ExponentialMovingAverageState(alpha=0.9, count=10, smoothed_average=1.35).count == 10


def test_exponential_moving_average_count_empty() -> None:
    assert ExponentialMovingAverageState().count == 0


def test_exponential_moving_average_clone() -> None:
    state = ExponentialMovingAverageState(alpha=0.9, count=10, smoothed_average=1.35)
    state_cloned = state.clone()
    assert state is not state_cloned
    assert state_cloned.equal(
        ExponentialMovingAverageState(alpha=0.9, count=10, smoothed_average=1.35)
    )


def test_exponential_moving_average_clone_empty() -> None:
    state = ExponentialMovingAverageState()
    state_cloned = state.clone()
    assert state is not state_cloned
    assert state_cloned.equal(ExponentialMovingAverageState())


def test_exponential_moving_average_equal_true() -> None:
    assert ExponentialMovingAverageState(alpha=0.9, count=10, smoothed_average=1.35).equal(
        ExponentialMovingAverageState(alpha=0.9, count=10, smoothed_average=1.35)
    )


def test_exponential_moving_average_equal_true_empty() -> None:
    assert ExponentialMovingAverageState().equal(ExponentialMovingAverageState())


def test_exponential_moving_average_equal_false_different_alpha() -> None:
    assert not ExponentialMovingAverageState(alpha=0.9, count=10, smoothed_average=1.35).equal(
        ExponentialMovingAverageState(alpha=0.95, count=10, smoothed_average=1.35)
    )


def test_exponential_moving_average_equal_false_different_count() -> None:
    assert not ExponentialMovingAverageState(alpha=0.9, count=10, smoothed_average=1.35).equal(
        ExponentialMovingAverageState(alpha=0.9, count=9, smoothed_average=1.35)
    )


def test_exponential_moving_average_equal_false_different_smoothed_average() -> None:
    assert not ExponentialMovingAverageState(alpha=0.9, count=10, smoothed_average=1.35).equal(
        ExponentialMovingAverageState(alpha=0.9, count=10, smoothed_average=2.35)
    )


def test_exponential_moving_average_equal_false_different_type() -> None:
    assert not ExponentialMovingAverageState(alpha=0.9, count=10, smoothed_average=1.35).equal(1)


def test_exponential_moving_average_load_state_dict() -> None:
    state = ExponentialMovingAverageState()
    state.load_state_dict({"alpha": 0.9, "count": 10, "smoothed_average": 1.35})
    assert state.equal(ExponentialMovingAverageState(alpha=0.9, count=10, smoothed_average=1.35))


def test_exponential_moving_average_reset() -> None:
    state = ExponentialMovingAverageState(alpha=0.9, count=10, smoothed_average=1.35)
    state.reset()
    assert state.equal(ExponentialMovingAverageState(alpha=0.9))


def test_exponential_moving_average_reset_empty() -> None:
    state = ExponentialMovingAverageState()
    state.reset()
    assert state.equal(ExponentialMovingAverageState())


def test_exponential_moving_average_state_dict() -> None:
    assert ExponentialMovingAverageState(
        alpha=0.9, count=10, smoothed_average=1.35
    ).state_dict() == {
        "alpha": 0.9,
        "count": 10,
        "smoothed_average": 1.35,
    }


def test_exponential_moving_average_state_dict_empty() -> None:
    assert ExponentialMovingAverageState().state_dict() == {
        "alpha": 0.98,
        "count": 0,
        "smoothed_average": 0.0,
    }


def test_exponential_moving_average_smoothed_average() -> None:
    assert ExponentialMovingAverageState(smoothed_average=1.35, count=1).smoothed_average() == 1.35


def test_exponential_moving_average_smoothed_average_empty() -> None:
    state = ExponentialMovingAverageState()
    with pytest.raises(EmptyStateError, match="The state is empty"):
        state.smoothed_average()


def test_exponential_moving_average_update() -> None:
    state = ExponentialMovingAverageState()
    state.update(4)
    state.update(2)
    state.equal(ExponentialMovingAverageState(alpha=0.98, count=2, smoothed_average=3.96))
