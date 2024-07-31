from __future__ import annotations

import math
from unittest.mock import Mock, patch

import pytest
import torch

from karbonn.distributed.ddp import SUM
from karbonn.utils.tracker import EmptyTrackerError, MeanTensorTracker

#######################################
#     Tests for MeanTensorTracker     #
#######################################


def test_mean_tensor_tracker_repr() -> None:
    assert repr(MeanTensorTracker(count=8, total=20.0)) == "MeanTensorTracker(count=8, total=20.0)"


def test_mean_tensor_tracker_str() -> None:
    assert str(MeanTensorTracker(count=8, total=20.0)) == "MeanTensorTracker(count=8, total=20.0)"


def test_mean_tensor_tracker_str_empty() -> None:
    assert str(MeanTensorTracker()) == "MeanTensorTracker(count=0, total=0.0)"


def test_mean_tensor_tracker_count() -> None:
    assert MeanTensorTracker(count=8).count == 8


def test_mean_tensor_tracker_count_empty() -> None:
    assert MeanTensorTracker().count == 0


def test_mean_tensor_tracker_total() -> None:
    assert MeanTensorTracker(total=12.0).total == 12.0


def test_mean_tensor_tracker_total_empty() -> None:
    assert MeanTensorTracker().total == 0


def test_mean_tensor_tracker_reset() -> None:
    tracker = MeanTensorTracker(count=8, total=20.0)
    tracker.reset()
    assert tracker.equal(MeanTensorTracker())


def test_mean_tensor_tracker_update() -> None:
    tracker = MeanTensorTracker()
    tracker.update(torch.arange(6))
    tracker.update(torch.tensor([4.0, 1.0]))
    assert tracker.equal(MeanTensorTracker(count=8, total=20.0))


def test_mean_tensor_tracker_update_1d() -> None:
    tracker = MeanTensorTracker()
    tracker.update(torch.arange(6))
    assert tracker.equal(MeanTensorTracker(count=6, total=15))


def test_mean_tensor_tracker_update_2d() -> None:
    tracker = MeanTensorTracker()
    tracker.update(torch.arange(6).view(2, 3))
    assert tracker.equal(MeanTensorTracker(count=6, total=15))


def test_mean_tensor_tracker_update_3d() -> None:
    tracker = MeanTensorTracker()
    tracker.update(torch.ones(2, 3, 4))
    assert tracker.equal(MeanTensorTracker(count=24, total=24.0))


def test_mean_tensor_tracker_update_float() -> None:
    tracker = MeanTensorTracker()
    tracker.update(torch.tensor([4.0, 1.0], dtype=torch.float))
    assert tracker.equal(MeanTensorTracker(count=2, total=5.0))


def test_mean_tensor_tracker_update_long() -> None:
    tracker = MeanTensorTracker()
    tracker.update(torch.tensor([4, 1], dtype=torch.long))
    assert tracker.equal(MeanTensorTracker(count=2, total=5))


def test_mean_tensor_tracker_update_nan() -> None:
    tracker = MeanTensorTracker()
    tracker.update(torch.tensor(float("NaN")))
    assert math.isnan(tracker.sum())
    assert tracker.count == 1


def test_mean_tensor_tracker_update_inf() -> None:
    tracker = MeanTensorTracker()
    tracker.update(torch.tensor(float("inf")))
    assert tracker.equal(MeanTensorTracker(count=1, total=float("inf")))


def test_mean_tensor_tracker_average() -> None:
    assert MeanTensorTracker(count=8, total=20.0).average() == 2.5


def test_mean_tensor_tracker_average_empty() -> None:
    tracker = MeanTensorTracker()
    with pytest.raises(EmptyTrackerError, match="The tracker is empty"):
        tracker.average()


def test_mean_tensor_tracker_mean() -> None:
    assert MeanTensorTracker(count=8, total=20.0).mean() == 2.5


def test_mean_tensor_tracker_mean_empty() -> None:
    tracker = MeanTensorTracker()
    with pytest.raises(EmptyTrackerError, match="The tracker is empty"):
        tracker.mean()


def test_mean_tensor_tracker_sum_int() -> None:
    total = MeanTensorTracker(count=8, total=22).sum()
    assert total == 22
    assert isinstance(total, int)


def test_mean_tensor_tracker_sum_float() -> None:
    total = MeanTensorTracker(count=8, total=20.0).sum()
    assert total == 20.0
    assert isinstance(total, float)


def test_mean_tensor_tracker_sum_empty() -> None:
    tracker = MeanTensorTracker()
    with pytest.raises(EmptyTrackerError, match="The tracker is empty"):
        tracker.sum()


def test_mean_tensor_tracker_all_reduce() -> None:
    tracker = MeanTensorTracker(count=10, total=122.0)
    tracker_reduced = tracker.all_reduce()
    assert tracker_reduced is not tracker
    assert tracker.equal(MeanTensorTracker(count=10, total=122.0))
    assert tracker_reduced.equal(MeanTensorTracker(count=10, total=122.0))


def test_mean_tensor_tracker_all_reduce_empty() -> None:
    tracker = MeanTensorTracker()
    tracker_reduced = tracker.all_reduce()
    assert tracker_reduced is not tracker
    assert tracker.equal(MeanTensorTracker())
    assert tracker_reduced.equal(MeanTensorTracker())


def test_mean_tensor_tracker_all_reduce_sum_reduce() -> None:
    tracker = MeanTensorTracker(count=10, total=122.0)
    reduce_mock = Mock(side_effect=lambda variable, op: variable + 1)  # noqa: ARG005
    with patch("karbonn.utils.tracker.tensor.sync_reduce", reduce_mock):
        tracker_reduced = tracker.all_reduce()
        assert tracker.equal(MeanTensorTracker(count=10, total=122.0))
        assert tracker_reduced.equal(MeanTensorTracker(count=11, total=123.0))
        assert reduce_mock.call_args_list == [((10, SUM), {}), ((122.0, SUM), {})]


def test_mean_tensor_tracker_clone() -> None:
    tracker = MeanTensorTracker(count=10, total=122.0)
    tracker_cloned = tracker.clone()
    assert tracker_cloned is not tracker
    assert tracker.equal(MeanTensorTracker(count=10, total=122.0))
    assert tracker_cloned.equal(MeanTensorTracker(count=10, total=122.0))


def test_mean_tensor_tracker_clone_empty() -> None:
    tracker = MeanTensorTracker()
    tracker_cloned = tracker.clone()
    assert tracker_cloned is not tracker
    assert tracker.equal(MeanTensorTracker())
    assert tracker_cloned.equal(MeanTensorTracker())


def test_mean_tensor_tracker_equal_true() -> None:
    assert MeanTensorTracker(total=122.0, count=10).equal(MeanTensorTracker(total=122.0, count=10))


def test_mean_tensor_tracker_equal_true_empty() -> None:
    assert MeanTensorTracker().equal(MeanTensorTracker())


def test_mean_tensor_tracker_equal_false_different_count() -> None:
    assert not MeanTensorTracker(total=122.0, count=10).equal(
        MeanTensorTracker(total=122.0, count=9)
    )


def test_mean_tensor_tracker_equal_false_different_total() -> None:
    assert not MeanTensorTracker(total=122.0, count=10).equal(
        MeanTensorTracker(total=12.0, count=10)
    )


def test_mean_tensor_tracker_equal_false_different_type() -> None:
    assert not MeanTensorTracker(total=122.0, count=10).equal(1)


def test_mean_tensor_tracker_merge() -> None:
    tracker = MeanTensorTracker(total=122.0, count=10)
    tracker_merged = tracker.merge(
        [
            MeanTensorTracker(total=1.0, count=4),
            MeanTensorTracker(),
            MeanTensorTracker(total=-2.0, count=2),
        ]
    )
    assert tracker.equal(MeanTensorTracker(total=122.0, count=10))
    assert tracker_merged.equal(MeanTensorTracker(total=121.0, count=16))


def test_mean_tensor_tracker_merge_() -> None:
    tracker = MeanTensorTracker(total=122.0, count=10)
    tracker.merge_(
        [
            MeanTensorTracker(total=1.0, count=4),
            MeanTensorTracker(),
            MeanTensorTracker(total=-2.0, count=2),
        ]
    )
    assert tracker.equal(MeanTensorTracker(total=121.0, count=16))


def test_mean_tensor_tracker_load_state_dict() -> None:
    tracker = MeanTensorTracker()
    tracker.load_state_dict({"count": 10, "total": 122.0})
    assert tracker.equal(MeanTensorTracker(count=10, total=122.0))
    assert tracker.count == 10


def test_mean_tensor_tracker_state_dict() -> None:
    assert MeanTensorTracker(count=6, total=15.0).state_dict() == {"count": 6, "total": 15.0}


def test_mean_tensor_tracker_state_dict_empty() -> None:
    assert MeanTensorTracker().state_dict() == {
        "count": 0,
        "total": 0,
    }
