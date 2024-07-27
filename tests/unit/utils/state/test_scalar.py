from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from karbonn.utils.state import EmptyStateError, ScalarState

#################################
#     Tests for ScalarState     #
#################################


def test_scalar_state_repr() -> None:
    assert (
        repr(ScalarState())
        == "ScalarState(count=0.0, total=0.0, min_value=inf, max_value=-inf, max_size=100)"
    )


@patch("karbonn.utils.state.ScalarState.std", lambda *args: 1.5)  # noqa: ARG005
def test_scalar_state_str() -> None:
    assert str(
        ScalarState(total=6.0, count=2, max_value=4.0, min_value=2.0, values=(4.0, 2.0))
    ) == (
        """ScalarState(
  (average): 3.0
  (count): 2.0
  (max): 4.0
  (median): 2.0
  (min): 2.0
  (std): 1.5
  (sum): 6.0
)"""
    )


def test_scalar_state_str_empty() -> None:
    assert str(ScalarState()) == (
        """ScalarState(
  (average): N/A (empty)
  (count): 0.0
  (max): N/A (empty)
  (median): N/A (empty)
  (min): N/A (empty)
  (std): N/A (empty)
  (sum): N/A (empty)
)"""
    )


def test_scalar_state_count() -> None:
    assert (
        ScalarState(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        ).count
        == 10
    )


def test_scalar_state_count_empty() -> None:
    assert ScalarState().count == 0


def test_scalar_state_total() -> None:
    assert (
        ScalarState(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        ).total
        == 122.0
    )


def test_scalar_state_total_empty() -> None:
    assert ScalarState().total == 0


def test_scalar_state_values() -> None:
    assert ScalarState(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).values == (1.0, 3.0, 5.0, 4.0, 2.0)


def test_scalar_state_values_empty() -> None:
    assert ScalarState().values == ()


def test_scalar_state_average() -> None:
    assert (
        ScalarState(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        ).average()
        == 12.2
    )


def test_scalar_state_average_empty() -> None:
    state = ScalarState()
    with pytest.raises(EmptyStateError, match="The state is empty"):
        state.average()


def test_scalar_state_equal_true() -> None:
    assert ScalarState(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).equal(
        ScalarState(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_state_equal_true_empty() -> None:
    assert ScalarState().equal(ScalarState())


def test_scalar_state_equal_false_different_count() -> None:
    assert not ScalarState(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).equal(
        ScalarState(
            total=122.0, count=9, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_state_equal_false_different_total() -> None:
    assert not ScalarState(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).equal(
        ScalarState(
            total=12.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_state_equal_false_different_max_value() -> None:
    assert not ScalarState(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).equal(
        ScalarState(
            total=122.0, count=10, max_value=16.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_state_equal_false_different_min_value() -> None:
    assert not ScalarState(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).equal(
        ScalarState(
            total=122.0, count=10, max_value=6.0, min_value=-12.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_state_equal_false_different_values() -> None:
    assert not ScalarState(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).equal(
        ScalarState(total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1, 2, 3, 4, 6))
    )


def test_scalar_state_equal_false_different_type() -> None:
    assert not ScalarState(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).equal(1)


def test_scalar_state_load_state_dict() -> None:
    state = ScalarState()
    state.load_state_dict(
        {
            "count": 10,
            "total": 122.0,
            "values": (1.0, 3.0, 5.0, 4.0, 2.0),
            "max_value": 6.0,
            "min_value": -2.0,
        }
    )
    assert state.equal(
        ScalarState(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_state_max() -> None:
    assert (
        ScalarState(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        ).max()
        == 6.0
    )


def test_scalar_state_max_empty() -> None:
    state = ScalarState()
    with pytest.raises(EmptyStateError, match="The state is empty"):
        state.max()


def test_scalar_state_median() -> None:
    assert (
        ScalarState(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        ).median()
        == 3.0
    )


def test_scalar_state_median_empty() -> None:
    state = ScalarState()
    with pytest.raises(EmptyStateError, match="The state is empty"):
        state.median()


def test_scalar_state_merge() -> None:
    state = ScalarState(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    )
    state_merged = state.merge(
        [
            ScalarState(total=12.0, count=5, max_value=8.0, min_value=-1.0, values=(1.0, 3.0)),
            ScalarState(),
            ScalarState(total=-5.0, count=2, max_value=2.0, min_value=-5.0, values=(-5.0, 2.0)),
        ]
    )
    assert state.equal(
        ScalarState(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )
    assert state_merged.equal(
        ScalarState(
            total=129.0, count=17, max_value=8.0, min_value=-5.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_state_merge_() -> None:
    state = ScalarState(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    )
    state.merge_(
        [
            ScalarState(total=12.0, count=5, max_value=8.0, min_value=-1.0, values=(1.0, 3.0)),
            ScalarState(),
            ScalarState(total=-5.0, count=2, max_value=2.0, min_value=-5.0, values=(-5.0, 2.0)),
        ]
    )
    assert state.equal(
        ScalarState(
            total=129.0, count=17, max_value=8.0, min_value=-5.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        )
    )


def test_scalar_state_min() -> None:
    assert (
        ScalarState(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        ).min()
        == -2.0
    )


def test_scalar_state_min_empty() -> None:
    state = ScalarState()
    with pytest.raises(EmptyStateError, match="The state is empty"):
        state.min()


def test_scalar_state_reset() -> None:
    state = ScalarState(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    )
    state.reset()
    assert state.equal(ScalarState())


def test_scalar_state_reset_empty() -> None:
    state = ScalarState()
    state.reset()
    assert state.equal(ScalarState())


def test_scalar_state_state_dict() -> None:
    assert ScalarState(
        total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
    ).state_dict() == {
        "count": 10,
        "total": 122.0,
        "values": (1.0, 3.0, 5.0, 4.0, 2.0),
        "max_value": 6.0,
        "min_value": -2.0,
    }


def test_scalar_state_state_dict_empty() -> None:
    assert ScalarState().state_dict() == {
        "count": 0,
        "total": 0.0,
        "values": (),
        "max_value": -float("inf"),
        "min_value": float("inf"),
    }


def test_scalar_state_std() -> None:
    assert (
        ScalarState(
            total=10.0, count=10, max_value=1.0, min_value=1.0, values=(1.0, 1.0, 1.0, 1.0, 1.0)
        ).std()
        == 0.0
    )


def test_scalar_state_std_empty() -> None:
    state = ScalarState()
    with pytest.raises(EmptyStateError, match="The state is empty"):
        state.std()


def test_scalar_state_sum() -> None:
    assert (
        ScalarState(
            total=122.0, count=10, max_value=6.0, min_value=-2.0, values=(1.0, 3.0, 5.0, 4.0, 2.0)
        ).sum()
        == 122.0
    )


def test_scalar_state_sum_empty() -> None:
    state = ScalarState()
    with pytest.raises(EmptyStateError, match="The state is empty"):
        state.sum()


def test_scalar_state_update_4() -> None:
    state = ScalarState()
    state.update(4)
    assert state.equal(ScalarState(count=1, total=4.0, min_value=4.0, max_value=4.0, values=(4.0,)))


def test_scalar_state_update_4_and_2() -> None:
    state = ScalarState()
    state.update(4)
    state.update(2)
    assert state.equal(
        ScalarState(count=2, total=6.0, min_value=2.0, max_value=4.0, values=(4.0, 2.0))
    )


def test_scalar_state_update_max_window_size_3() -> None:
    state = ScalarState(max_size=3)
    state.update(0)
    state.update(3)
    state.update(1)
    state.update(4)
    assert state.equal(
        ScalarState(count=4, total=8.0, min_value=0.0, max_value=4.0, values=(3.0, 1.0, 4.0))
    )


def test_scalar_state_update_nan() -> None:
    state = ScalarState()
    state.update(float("NaN"))
    assert math.isnan(state.total)
    assert state.count == 1


def test_scalar_state_update_inf() -> None:
    state = ScalarState()
    state.update(float("inf"))
    assert state.equal(
        ScalarState(
            count=1,
            total=float("inf"),
            min_value=float("inf"),
            max_value=float("inf"),
            values=(float("inf"),),
        )
    )


@pytest.mark.parametrize("values", [[3, 1, 2], (3, 1, 2), (3.0, 1.0, 2.0)])
def test_scalar_state_update_sequence(values: list[float] | tuple[float, ...]) -> None:
    state = ScalarState()
    state.update_sequence(values)
    assert state.equal(
        ScalarState(count=3, total=6.0, min_value=1.0, max_value=3.0, values=(3.0, 1.0, 2.0))
    )
