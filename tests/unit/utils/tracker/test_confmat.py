from __future__ import annotations

import math
from unittest.mock import Mock, patch

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal
from minrecord import MaxScalarRecord, MinScalarRecord

from karbonn.distributed.ddp import SUM
from karbonn.testing import tabulate_available
from karbonn.utils.tracker import EmptyTrackerError
from karbonn.utils.tracker.confmat import (
    BinaryConfusionMatrixTracker,
    MulticlassConfusionMatrixTracker,
    check_confusion_matrix,
    check_op_compatibility_binary,
    check_op_compatibility_multiclass,
    str_binary_confusion_matrix,
)

##################################################
#     Tests for BinaryConfusionMatrixTracker     #
##################################################


def test_binary_confusion_matrix_tracker_repr() -> None:
    assert isinstance(repr(BinaryConfusionMatrixTracker()), str)


@tabulate_available
def test_binary_confusion_matrix_tracker_repr_tabulate() -> None:
    assert repr(BinaryConfusionMatrixTracker()) == (
        "┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
        "┃                     ┃ predicted negative (0) ┃ predicted positive (1) ┃\n"
        "┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫\n"
        "┃ actual negative (0) ┃ [TN]  0                ┃ [FP]  0                ┃\n"
        "┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫\n"
        "┃ actual positive (1) ┃ [FN]  0                ┃ [TP]  0                ┃\n"
        "┗━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
        "count=0"
    )


def test_binary_confusion_matrix_tracker_str() -> None:
    assert str(BinaryConfusionMatrixTracker()).startswith("BinaryConfusionMatrixTracker(")


def test_binary_confusion_matrix_tracker_init_default() -> None:
    meter = BinaryConfusionMatrixTracker()
    assert objects_are_equal(meter.matrix, torch.zeros(2, 2, dtype=torch.long))
    assert meter.count == 0


def test_binary_confusion_matrix_tracker_init() -> None:
    meter = BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]]))
    assert objects_are_equal(meter.matrix, torch.tensor([[3, 2], [1, 4]]))
    assert meter.count == 10


def test_binary_confusion_matrix_tracker_init_incorrect_ndim() -> None:
    with pytest.raises(ValueError, match="Incorrect shape."):
        BinaryConfusionMatrixTracker(torch.zeros(3))


def test_binary_confusion_matrix_tracker_init_incorrect_shape() -> None:
    with pytest.raises(ValueError, match="Incorrect shape."):
        BinaryConfusionMatrixTracker(torch.zeros(3, 5))


def test_binary_confusion_matrix_tracker_init_incorrect_dtype() -> None:
    with pytest.raises(ValueError, match="Incorrect matrix data type."):
        BinaryConfusionMatrixTracker(torch.zeros(2, 2, dtype=torch.float))


def test_binary_confusion_matrix_tracker_init_negative_value() -> None:
    with pytest.raises(ValueError, match="Incorrect matrix values."):
        BinaryConfusionMatrixTracker(torch.tensor([[0, 0], [-1, 0]]))


def test_binary_confusion_matrix_tracker_num_classes() -> None:
    assert BinaryConfusionMatrixTracker().num_classes == 2


def test_binary_confusion_matrix_tracker_all_reduce() -> None:
    meter = BinaryConfusionMatrixTracker(torch.ones(2, 2, dtype=torch.long)).all_reduce()
    assert meter.matrix.equal(torch.ones(2, 2, dtype=torch.long))
    assert meter.count == 4


def test_binary_confusion_matrix_tracker_all_reduce_sum_reduce() -> None:
    meter = BinaryConfusionMatrixTracker(torch.ones(2, 2, dtype=torch.long))
    reduce_mock = Mock(return_value=torch.ones(2, 2, dtype=torch.long))
    with patch("karbonn.utils.tracker.confmat.sync_reduce", reduce_mock):
        meter.all_reduce()
        assert objects_are_equal(
            reduce_mock.call_args.args, (torch.ones(2, 2, dtype=torch.long), SUM)
        )


def test_binary_confusion_matrix_tracker_clone() -> None:
    meter = BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]], dtype=torch.long))
    meter_cloned = meter.clone()
    assert meter is not meter_cloned
    assert meter.equal(meter_cloned)


def test_binary_confusion_matrix_tracker_equal_true() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).equal(
        BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]]))
    )


def test_binary_confusion_matrix_tracker_equal_false_different_values() -> None:
    assert not BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).equal(
        BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [0, 4]]))
    )


def test_binary_confusion_matrix_tracker_equal_false_different_type() -> None:
    assert not BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).equal(42)


def test_binary_confusion_matrix_tracker_get_normalized_matrix_normalization_true() -> None:
    assert (
        BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]]))
        .get_normalized_matrix(normalization="true")
        .equal(torch.tensor([[0.6, 0.4], [0.2, 0.8]], dtype=torch.float))
    )


def test_binary_confusion_matrix_tracker_get_normalized_matrix_normalization_true_empty() -> None:
    assert (
        BinaryConfusionMatrixTracker()
        .get_normalized_matrix(normalization="true")
        .equal(torch.zeros(2, 2, dtype=torch.float))
    )


def test_binary_confusion_matrix_tracker_get_normalized_matrix_normalization_pred() -> None:
    assert (
        BinaryConfusionMatrixTracker(torch.tensor([[3, 6], [1, 4]]))
        .get_normalized_matrix(normalization="pred")
        .equal(torch.tensor([[0.75, 0.6], [0.25, 0.4]], dtype=torch.float))
    )


def test_binary_confusion_matrix_tracker_get_normalized_matrix_normalization_pred_empty() -> None:
    assert (
        BinaryConfusionMatrixTracker()
        .get_normalized_matrix(normalization="pred")
        .equal(torch.zeros(2, 2, dtype=torch.float))
    )


def test_binary_confusion_matrix_tracker_get_normalized_matrix_normalization_all() -> None:
    assert (
        BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]]))
        .get_normalized_matrix(normalization="all")
        .equal(torch.tensor([[0.3, 0.2], [0.1, 0.4]], dtype=torch.float))
    )


def test_binary_confusion_matrix_tracker_get_normalized_matrix_normalization_all_empty() -> None:
    assert (
        BinaryConfusionMatrixTracker()
        .get_normalized_matrix(normalization="all")
        .equal(torch.zeros(2, 2, dtype=torch.float))
    )


def test_binary_confusion_matrix_tracker_get_normalized_matrix_incorrect_normalization() -> None:
    with pytest.raises(ValueError, match="Incorrect normalization: incorrect"):
        BinaryConfusionMatrixTracker().get_normalized_matrix(normalization="incorrect")


def test_binary_confusion_matrix_tracker_reset() -> None:
    meter = BinaryConfusionMatrixTracker(torch.ones(2, 2, dtype=torch.long))
    meter.reset()
    assert meter.matrix.equal(torch.zeros(2, 2, dtype=torch.long))
    assert meter.count == 0


def test_binary_confusion_matrix_tracker_sync_update_matrix() -> None:
    with patch(
        "karbonn.utils.tracker.confmat.sync_reduce",
        lambda variable, op: variable.mul_(4),  # noqa: ARG005
    ):
        meter = BinaryConfusionMatrixTracker(torch.ones(2, 2, dtype=torch.long)).all_reduce()
    assert meter.matrix.equal(torch.ones(2, 2, dtype=torch.long).mul(4))
    assert meter.count == 16


def test_binary_confusion_matrix_tracker_update() -> None:
    meter = BinaryConfusionMatrixTracker()
    meter.update(
        prediction=torch.tensor([0, 1, 1, 0, 0, 1], dtype=torch.long),
        target=torch.tensor([1, 1, 1, 0, 0, 1], dtype=torch.long),
    )
    assert meter.matrix.equal(torch.tensor([[2, 0], [1, 3]], dtype=torch.long))


def test_binary_confusion_matrix_tracker_update_2() -> None:
    meter = BinaryConfusionMatrixTracker()
    meter.update(
        prediction=torch.tensor([0, 1, 1, 0, 0, 1], dtype=torch.long),
        target=torch.tensor([1, 1, 1, 0, 0, 1], dtype=torch.long),
    )
    meter.update(
        prediction=torch.tensor([1, 1], dtype=torch.long),
        target=torch.tensor([0, 0], dtype=torch.long),
    )
    assert meter.matrix.equal(torch.tensor([[2, 2], [1, 3]], dtype=torch.long))


def test_binary_confusion_matrix_tracker_from_predictions() -> None:
    meter = BinaryConfusionMatrixTracker.from_predictions(
        prediction=torch.tensor([0, 1, 1, 0, 0, 1], dtype=torch.long),
        target=torch.tensor([1, 1, 1, 0, 0, 1], dtype=torch.long),
    )
    assert meter.matrix.equal(torch.tensor([[2, 0], [1, 3]], dtype=torch.long))
    assert meter.true_positive == 3
    assert meter.true_negative == 2
    assert meter.false_negative == 1
    assert meter.false_positive == 0


# **************************
# *     Transformation     *
# **************************


def test_binary_confusion_matrix_tracker__add__() -> None:
    assert (
        BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]]))
        + BinaryConfusionMatrixTracker(torch.tensor([[1, 0], [7, 2]]))
    ).equal(BinaryConfusionMatrixTracker(torch.tensor([[4, 2], [8, 6]])))


def test_binary_confusion_matrix_tracker__iadd__() -> None:
    meter = BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]]))
    meter += BinaryConfusionMatrixTracker(torch.tensor([[1, 0], [7, 2]]))
    assert meter.equal(BinaryConfusionMatrixTracker(torch.tensor([[4, 2], [8, 6]])))


def test_binary_confusion_matrix_tracker__sub__() -> None:
    assert (
        BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]]))
        - BinaryConfusionMatrixTracker(torch.tensor([[1, 0], [1, 2]]))
    ).equal(BinaryConfusionMatrixTracker(torch.tensor([[2, 2], [0, 2]])))


def test_binary_confusion_matrix_tracker_add() -> None:
    meter = BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).add(
        BinaryConfusionMatrixTracker(torch.tensor([[1, 0], [7, 2]]))
    )
    assert meter.equal(BinaryConfusionMatrixTracker(torch.tensor([[4, 2], [8, 6]])))
    assert meter.count == 20


def test_binary_confusion_matrix_tracker_add_() -> None:
    meter = BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]]))
    meter.add_(BinaryConfusionMatrixTracker(torch.tensor([[1, 0], [7, 2]])))
    assert meter.equal(BinaryConfusionMatrixTracker(torch.tensor([[4, 2], [8, 6]])))
    assert meter.count == 20


def test_binary_confusion_matrix_tracker_merge() -> None:
    meter = BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]]))
    meter_merged = meter.merge(
        [
            BinaryConfusionMatrixTracker(torch.tensor([[1, 0], [7, 2]])),
            BinaryConfusionMatrixTracker(torch.tensor([[1, 0], [0, 1]])),
        ]
    )
    assert meter.equal(BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])))
    assert meter.count == 10
    assert meter_merged.equal(BinaryConfusionMatrixTracker(torch.tensor([[5, 2], [8, 7]])))
    assert meter_merged.count == 22


def test_binary_confusion_matrix_tracker_merge_() -> None:
    meter = BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]]))
    meter.merge_(
        [
            BinaryConfusionMatrixTracker(torch.tensor([[1, 0], [7, 2]])),
            BinaryConfusionMatrixTracker(torch.tensor([[1, 0], [0, 1]])),
        ]
    )
    assert meter.equal(BinaryConfusionMatrixTracker(torch.tensor([[5, 2], [8, 7]])))
    assert meter.count == 22


def test_binary_confusion_matrix_tracker_sub() -> None:
    meter = BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).sub(
        BinaryConfusionMatrixTracker(torch.tensor([[1, 0], [1, 2]]))
    )
    assert meter.equal(BinaryConfusionMatrixTracker(torch.tensor([[2, 2], [0, 2]])))
    assert meter.count == 6


# *******************
# *     Metrics     *
# *******************


def test_binary_confusion_matrix_tracker_false_negative() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).false_negative == 1


def test_binary_confusion_matrix_tracker_false_positive() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).false_positive == 2


def test_binary_confusion_matrix_tracker_negative() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 5], [1, 4]])).negative == 8


def test_binary_confusion_matrix_tracker_positive() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 5], [1, 4]])).positive == 5


def test_binary_confusion_matrix_tracker_predictive_negative() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).predictive_negative == 4


def test_binary_confusion_matrix_tracker_predictive_positive() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).predictive_positive == 6


def test_binary_confusion_matrix_tracker_true_negative() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).true_negative == 3


def test_binary_confusion_matrix_tracker_true_positive() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).true_positive == 4


def test_binary_confusion_matrix_tracker_accuracy() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).accuracy() == 0.7


def test_binary_confusion_matrix_tracker_accuracy_imbalanced() -> None:
    assert math.isclose(
        BinaryConfusionMatrixTracker(torch.tensor([[30, 2], [1, 4]])).accuracy(), 0.918918918918919
    )


def test_binary_confusion_matrix_tracker_accuracy_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match="It is not possible to compute the accuracy because the confusion matrix is empty",
    ):
        BinaryConfusionMatrixTracker().accuracy()


def test_binary_confusion_matrix_tracker_balanced_accuracy() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).balanced_accuracy() == 0.7


def test_binary_confusion_matrix_tracker_balanced_accuracy_imbalanced() -> None:
    assert (
        BinaryConfusionMatrixTracker(torch.tensor([[30, 2], [1, 4]])).balanced_accuracy() == 0.86875
    )


def test_binary_confusion_matrix_tracker_balanced_accuracy_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match=(
            "It is not possible to compute the balanced accuracy because "
            "the confusion matrix is empty"
        ),
    ):
        BinaryConfusionMatrixTracker().balanced_accuracy()


def test_binary_confusion_matrix_tracker_f_beta_score_1() -> None:
    assert math.isclose(
        BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).f_beta_score(),
        0.7272727272727273,
    )


def test_binary_confusion_matrix_tracker_f_beta_score_2() -> None:
    assert math.isclose(
        BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).f_beta_score(beta=2),
        0.7692307692307693,
    )


def test_binary_confusion_matrix_tracker_f_beta_score_0_5() -> None:
    assert math.isclose(
        BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).f_beta_score(beta=0.5),
        0.6896551724137931,
    )


def test_binary_confusion_matrix_tracker_f_beta_score_1_true_negative_only() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 0], [0, 0]])).f_beta_score() == 0.0


def test_binary_confusion_matrix_tracker_f_beta_score_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match=(
            "It is not possible to compute the F-beta score because "
            "the confusion matrix is empty"
        ),
    ):
        BinaryConfusionMatrixTracker().f_beta_score()


def test_binary_confusion_matrix_tracker_false_negative_rate() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).false_negative_rate() == 0.2


def test_binary_confusion_matrix_tracker_false_negative_rate_zero_positive() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [0, 0]])).false_negative_rate() == 0.0


def test_binary_confusion_matrix_tracker_false_negative_rate_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match=(
            "It is not possible to compute the false negative rate because "
            "the confusion matrix is empty"
        ),
    ):
        BinaryConfusionMatrixTracker().false_negative_rate()


def test_binary_confusion_matrix_tracker_false_positive_rate() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).false_positive_rate() == 0.4


def test_binary_confusion_matrix_tracker_false_positive_rate_zero_negative() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[0, 0], [1, 4]])).false_positive_rate() == 0.0


def test_binary_confusion_matrix_tracker_false_positive_rate_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match=(
            "It is not possible to compute the false positive rate because "
            "the confusion matrix is empty"
        ),
    ):
        BinaryConfusionMatrixTracker().false_positive_rate()


def test_binary_confusion_matrix_tracker_jaccard_index() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 3], [1, 4]])).jaccard_index() == 0.5


def test_binary_confusion_matrix_tracker_jaccard_index_zero_true_positive() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 0], [0, 0]])).jaccard_index() == 0.0


def test_binary_confusion_matrix_tracker_jaccard_index_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match=(
            "It is not possible to compute the Jaccard index because "
            "the confusion matrix is empty"
        ),
    ):
        BinaryConfusionMatrixTracker().jaccard_index()


def test_binary_confusion_matrix_tracker_precision() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 4], [1, 4]])).precision() == 0.5


def test_binary_confusion_matrix_tracker_precision_zero_predictive_positive() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 0], [1, 0]])).precision() == 0.0


def test_binary_confusion_matrix_tracker_precision_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match="It is not possible to compute the precision because the confusion matrix is empty",
    ):
        BinaryConfusionMatrixTracker().precision()


def test_binary_confusion_matrix_tracker_recall() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).recall() == 0.8


def test_binary_confusion_matrix_tracker_recall_zero_positive() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [0, 0]])).recall() == 0.0


def test_binary_confusion_matrix_tracker_recall_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match="It is not possible to compute the recall because the confusion matrix is empty",
    ):
        BinaryConfusionMatrixTracker().recall()


def test_binary_confusion_matrix_tracker_true_negative_rate() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).true_negative_rate() == 0.6


def test_binary_confusion_matrix_tracker_true_negative_rate_zero_negative() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[0, 0], [1, 4]])).true_negative_rate() == 0.0


def test_binary_confusion_matrix_tracker_true_negative_rate_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match=(
            "It is not possible to compute the true negative rate because "
            "the confusion matrix is empty"
        ),
    ):
        BinaryConfusionMatrixTracker().true_negative_rate()


def test_binary_confusion_matrix_tracker_true_positive_rate() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).true_positive_rate() == 0.8


def test_binary_confusion_matrix_tracker_true_positive_rate_zero_positive() -> None:
    assert BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [0, 0]])).true_positive_rate() == 0.0


def test_binary_confusion_matrix_tracker_true_positive_rate_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match=(
            "It is not possible to compute the true positive rate because "
            "the confusion matrix is empty"
        ),
    ):
        BinaryConfusionMatrixTracker().true_positive_rate()


def test_binary_confusion_matrix_tracker_compute_metrics() -> None:
    assert objects_are_allclose(
        BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).compute_metrics(),
        {
            "accuracy": 0.7,
            "balanced_accuracy": 0.7,
            "false_negative": 1,
            "false_negative_rate": 0.2,
            "false_positive": 2,
            "false_positive_rate": 0.4,
            "jaccard_index": 0.5714285714285714,
            "count": 10,
            "precision": 0.6666666666666666,
            "recall": 0.8,
            "true_negative": 3,
            "true_negative_rate": 0.6,
            "true_positive": 4,
            "true_positive_rate": 0.8,
            "f1_score": 0.7272727272727273,
        },
    )


def test_binary_confusion_matrix_tracker_compute_metrics_betas() -> None:
    assert objects_are_allclose(
        BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).compute_metrics(betas=(1, 2)),
        {
            "accuracy": 0.7,
            "balanced_accuracy": 0.7,
            "false_negative": 1,
            "false_negative_rate": 0.2,
            "false_positive": 2,
            "false_positive_rate": 0.4,
            "jaccard_index": 0.5714285714285714,
            "count": 10,
            "precision": 0.6666666666666666,
            "recall": 0.8,
            "true_negative": 3,
            "true_negative_rate": 0.6,
            "true_positive": 4,
            "true_positive_rate": 0.8,
            "f1_score": 0.7272727272727273,
            "f2_score": 0.7692307692307693,
        },
    )


def test_binary_confusion_matrix_tracker_compute_metrics_prefix_suffix() -> None:
    assert objects_are_allclose(
        BinaryConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).compute_metrics(
            prefix="prefix_", suffix="_suffix"
        ),
        {
            "prefix_accuracy_suffix": 0.7,
            "prefix_balanced_accuracy_suffix": 0.7,
            "prefix_false_negative_suffix": 1,
            "prefix_false_negative_rate_suffix": 0.2,
            "prefix_false_positive_suffix": 2,
            "prefix_false_positive_rate_suffix": 0.4,
            "prefix_jaccard_index_suffix": 0.5714285714285714,
            "prefix_count_suffix": 10,
            "prefix_precision_suffix": 0.6666666666666666,
            "prefix_recall_suffix": 0.8,
            "prefix_true_negative_suffix": 3,
            "prefix_true_negative_rate_suffix": 0.6,
            "prefix_true_positive_suffix": 4,
            "prefix_true_positive_rate_suffix": 0.8,
            "prefix_f1_score_suffix": 0.7272727272727273,
        },
    )


def test_binary_confusion_matrix_tracker_compute_metrics_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match="It is not possible to compute the metrics because the confusion matrix is empty",
    ):
        BinaryConfusionMatrixTracker().compute_metrics()


def test_binary_confusion_matrix_tracker_get_records() -> None:
    assert objects_are_equal(
        BinaryConfusionMatrixTracker().get_records(),
        (
            MaxScalarRecord(name="accuracy"),
            MaxScalarRecord(name="balanced_accuracy"),
            MaxScalarRecord(name="jaccard_index"),
            MaxScalarRecord(name="precision"),
            MaxScalarRecord(name="recall"),
            MaxScalarRecord(name="true_negative_rate"),
            MaxScalarRecord(name="true_negative"),
            MaxScalarRecord(name="true_positive_rate"),
            MaxScalarRecord(name="true_positive"),
            MinScalarRecord(name="false_negative_rate"),
            MinScalarRecord(name="false_negative"),
            MinScalarRecord(name="false_positive_rate"),
            MinScalarRecord(name="false_positive"),
            MaxScalarRecord(name="f1_score"),
        ),
    )


def test_binary_confusion_matrix_tracker_get_records_prefix_suffix() -> None:
    assert objects_are_equal(
        BinaryConfusionMatrixTracker().get_records(prefix="prefix_", suffix="_suffix"),
        (
            MaxScalarRecord(name="prefix_accuracy_suffix"),
            MaxScalarRecord(name="prefix_balanced_accuracy_suffix"),
            MaxScalarRecord(name="prefix_jaccard_index_suffix"),
            MaxScalarRecord(name="prefix_precision_suffix"),
            MaxScalarRecord(name="prefix_recall_suffix"),
            MaxScalarRecord(name="prefix_true_negative_rate_suffix"),
            MaxScalarRecord(name="prefix_true_negative_suffix"),
            MaxScalarRecord(name="prefix_true_positive_rate_suffix"),
            MaxScalarRecord(name="prefix_true_positive_suffix"),
            MinScalarRecord(name="prefix_false_negative_rate_suffix"),
            MinScalarRecord(name="prefix_false_negative_suffix"),
            MinScalarRecord(name="prefix_false_positive_rate_suffix"),
            MinScalarRecord(name="prefix_false_positive_suffix"),
            MaxScalarRecord(name="prefix_f1_score_suffix"),
        ),
    )


def test_binary_confusion_matrix_tracker_get_records_betas() -> None:
    assert objects_are_equal(
        BinaryConfusionMatrixTracker().get_records(betas=(0.5, 1, 2)),
        (
            MaxScalarRecord(name="accuracy"),
            MaxScalarRecord(name="balanced_accuracy"),
            MaxScalarRecord(name="jaccard_index"),
            MaxScalarRecord(name="precision"),
            MaxScalarRecord(name="recall"),
            MaxScalarRecord(name="true_negative_rate"),
            MaxScalarRecord(name="true_negative"),
            MaxScalarRecord(name="true_positive_rate"),
            MaxScalarRecord(name="true_positive"),
            MinScalarRecord(name="false_negative_rate"),
            MinScalarRecord(name="false_negative"),
            MinScalarRecord(name="false_positive_rate"),
            MinScalarRecord(name="false_positive"),
            MaxScalarRecord(name="f0.5_score"),
            MaxScalarRecord(name="f1_score"),
            MaxScalarRecord(name="f2_score"),
        ),
    )


######################################################
#     Tests for MulticlassConfusionMatrixTracker     #
######################################################


def test_multiclass_confusion_matrix_repr() -> None:
    assert repr(MulticlassConfusionMatrixTracker.from_num_classes(num_classes=5)).startswith(
        "MulticlassConfusionMatrixTracker("
    )


def test_multiclass_confusion_matrix_str() -> None:
    assert str(MulticlassConfusionMatrixTracker.from_num_classes(num_classes=5)).startswith(
        "MulticlassConfusionMatrixTracker("
    )


def test_multiclass_confusion_matrix_init_incorrect_ndim() -> None:
    with pytest.raises(ValueError, match="Incorrect matrix dimensions."):
        MulticlassConfusionMatrixTracker(torch.zeros(3))


def test_multiclass_confusion_matrix_init_incorrect_shape() -> None:
    with pytest.raises(ValueError, match="Incorrect matrix shape."):
        MulticlassConfusionMatrixTracker(torch.zeros(3, 5))


def test_multiclass_confusion_matrix_init_incorrect_dtype() -> None:
    with pytest.raises(ValueError, match="Incorrect matrix data type."):
        MulticlassConfusionMatrixTracker(torch.zeros(3, 3, dtype=torch.float))


def test_multiclass_confusion_matrix_init_negative_value() -> None:
    with pytest.raises(ValueError, match="Incorrect matrix values."):
        MulticlassConfusionMatrixTracker(torch.tensor([[0, 0], [-1, 0]]))


@pytest.mark.parametrize("num_classes", [2, 5])
def test_multiclass_confusion_matrix_num_classes(num_classes: int) -> None:
    assert (
        MulticlassConfusionMatrixTracker.from_num_classes(num_classes=num_classes).num_classes
        == num_classes
    )


def test_multiclass_confusion_matrix_all_reduce() -> None:
    meter = MulticlassConfusionMatrixTracker(
        torch.tensor([[2, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=torch.long)
    ).all_reduce()
    assert meter.matrix.equal(torch.tensor([[2, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=torch.long))
    assert meter.count == 6


def test_multiclass_confusion_matrix_all_reduce_sum_reduce() -> None:
    meter = MulticlassConfusionMatrixTracker(
        torch.tensor([[2, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=torch.long)
    )
    reduce_mock = Mock(
        return_value=torch.tensor([[2, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=torch.long)
    )
    with patch("karbonn.utils.tracker.confmat.sync_reduce", reduce_mock):
        meter.all_reduce()
        assert objects_are_equal(
            reduce_mock.call_args.args,
            (torch.tensor([[2, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=torch.long), SUM),
        )


def test_multiclass_confusion_matrix_auto_update_resize() -> None:
    meter = MulticlassConfusionMatrixTracker(
        torch.tensor([[2, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=torch.long)
    )
    meter.auto_update(torch.tensor([4, 2]), torch.tensor([4, 2]))
    assert meter.matrix.equal(
        torch.tensor(
            [
                [2, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 2, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
            ],
            dtype=torch.long,
        )
    )
    assert meter.count == 8


def test_multiclass_confusion_matrix_auto_update_no_resize() -> None:
    meter = MulticlassConfusionMatrixTracker(
        torch.tensor([[2, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=torch.long)
    )
    meter.auto_update(torch.tensor([1, 2]), torch.tensor([1, 2]))
    assert meter.matrix.equal(torch.tensor([[2, 1, 0], [0, 1, 0], [1, 1, 2]], dtype=torch.long))
    assert meter.count == 8


def test_multiclass_confusion_matrix_clone() -> None:
    meter = MulticlassConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]], dtype=torch.long))
    meter_cloned = meter.clone()
    assert meter is not meter_cloned
    assert meter.equal(meter_cloned)


def test_multiclass_confusion_matrix_equal_true() -> None:
    assert MulticlassConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).equal(
        MulticlassConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]]))
    )


def test_multiclass_confusion_matrix_equal_false_different_values() -> None:
    assert not MulticlassConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).equal(
        MulticlassConfusionMatrixTracker(torch.tensor([[3, 2], [0, 4]]))
    )


def test_multiclass_confusion_matrix_equal_false_different_type() -> None:
    assert not MulticlassConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).equal(42)


def test_multiclass_confusion_matrix_get_normalized_matrix_normalization_true() -> None:
    assert (
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        )
        .get_normalized_matrix(normalization="true")
        .equal(torch.tensor([[0.3, 0.2, 0.5], [0.2, 0.8, 0.0], [0.4, 0.2, 0.4]], dtype=torch.float))
    )


def test_multiclass_confusion_matrix_get_normalized_matrix_normalization_true_empty() -> None:
    assert (
        MulticlassConfusionMatrixTracker(torch.zeros(3, 3, dtype=torch.long))
        .get_normalized_matrix(normalization="true")
        .equal(torch.zeros(3, 3, dtype=torch.float))
    )


def test_multiclass_confusion_matrix_get_normalized_matrix_normalization_pred() -> None:
    assert (
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 1], [4, 2, 4]], dtype=torch.long)
        )
        .get_normalized_matrix(normalization="pred")
        .equal(
            torch.tensor(
                [[0.375, 0.25, 0.5], [0.125, 0.5, 0.1], [0.5, 0.25, 0.4]], dtype=torch.float
            )
        )
    )


def test_multiclass_confusion_matrix_get_normalized_matrix_normalization_pred_empty() -> None:
    assert (
        MulticlassConfusionMatrixTracker(torch.zeros(3, 3, dtype=torch.long))
        .get_normalized_matrix(normalization="pred")
        .equal(torch.zeros(3, 3, dtype=torch.float))
    )


def test_multiclass_confusion_matrix_get_normalized_matrix_normalization_all() -> None:
    assert (
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        )
        .get_normalized_matrix(normalization="all")
        .equal(
            torch.tensor(
                [[0.12, 0.08, 0.2], [0.04, 0.16, 0.0], [0.16, 0.08, 0.16]], dtype=torch.float
            )
        )
    )


def test_multiclass_confusion_matrix_get_normalized_matrix_normalization_all_empty() -> None:
    assert (
        MulticlassConfusionMatrixTracker(torch.zeros(3, 3, dtype=torch.long))
        .get_normalized_matrix(normalization="all")
        .equal(torch.zeros(3, 3, dtype=torch.float))
    )


def test_multiclass_confusion_matrix_get_normalized_matrix_incorrect_normalization() -> None:
    with pytest.raises(ValueError, match="Incorrect normalization: incorrect."):
        MulticlassConfusionMatrixTracker(torch.zeros(3, 3, dtype=torch.long)).get_normalized_matrix(
            normalization="incorrect"
        )


def test_multiclass_confusion_matrix_reset() -> None:
    meter = MulticlassConfusionMatrixTracker(torch.ones(3, 3, dtype=torch.long))
    meter.reset()
    assert meter.matrix.equal(torch.zeros(3, 3, dtype=torch.long))
    assert meter.count == 0


def test_multiclass_confusion_matrix_resize() -> None:
    meter = MulticlassConfusionMatrixTracker(
        torch.tensor([[2, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=torch.long)
    )
    meter.resize(num_classes=5)
    assert meter.matrix.equal(
        torch.tensor(
            [
                [2, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=torch.long,
        )
    )


def test_multiclass_confusion_matrix_resize_incorrect_num_classes() -> None:
    meter = MulticlassConfusionMatrixTracker(
        torch.tensor([[2, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=torch.long)
    )
    with pytest.raises(ValueError, match="Incorrect number of classes: 2."):
        meter.resize(num_classes=2)


def test_multiclass_confusion_matrix_update() -> None:
    meter = MulticlassConfusionMatrixTracker.from_num_classes(num_classes=3)
    meter.update(
        prediction=torch.tensor([0, 1, 2, 0, 0, 1], dtype=torch.long),
        target=torch.tensor([2, 2, 2, 0, 0, 0], dtype=torch.long),
    )
    assert meter.matrix.equal(torch.tensor([[2, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=torch.long))


def test_multiclass_confusion_matrix_update_2() -> None:
    meter = MulticlassConfusionMatrixTracker.from_num_classes(num_classes=3)
    meter.update(
        prediction=torch.tensor([0, 1, 2, 0, 0, 1], dtype=torch.long),
        target=torch.tensor([2, 2, 2, 0, 0, 0], dtype=torch.long),
    )
    meter.update(
        prediction=torch.tensor([1, 2, 0], dtype=torch.long),
        target=torch.tensor([2, 1, 0], dtype=torch.long),
    )
    assert meter.matrix.equal(torch.tensor([[3, 1, 0], [0, 0, 1], [1, 2, 1]], dtype=torch.long))


@pytest.mark.parametrize("num_classes", [2, 5])
def test_multiclass_confusion_matrix_from_num_classes(num_classes: int) -> None:
    assert MulticlassConfusionMatrixTracker.from_num_classes(num_classes=num_classes).matrix.equal(
        torch.zeros(num_classes, num_classes, dtype=torch.long)
    )


def test_multiclass_confusion_matrix_from_num_classes_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect number of classes."):
        MulticlassConfusionMatrixTracker.from_num_classes(num_classes=0)


def test_multiclass_confusion_matrix_from_predictions() -> None:
    assert MulticlassConfusionMatrixTracker.from_predictions(
        prediction=torch.tensor([0, 1, 2, 0, 0, 1], dtype=torch.long),
        target=torch.tensor([2, 2, 2, 0, 0, 0], dtype=torch.long),
    ).matrix.equal(torch.tensor([[2, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=torch.long))


# **************************
# *     Transformation     *
# **************************


def test_multiclass_confusion_matrix__add__() -> None:
    assert (
        MulticlassConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]]))
        + MulticlassConfusionMatrixTracker(torch.tensor([[1, 0], [7, 2]]))
    ).equal(MulticlassConfusionMatrixTracker(torch.tensor([[4, 2], [8, 6]])))


def test_multiclass_confusion_matrix__iadd__() -> None:
    meter = MulticlassConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]]))
    meter += MulticlassConfusionMatrixTracker(torch.tensor([[1, 0], [7, 2]]))
    assert meter.equal(MulticlassConfusionMatrixTracker(torch.tensor([[4, 2], [8, 6]])))


def test_multiclass_confusion_matrix__sub__() -> None:
    assert (
        MulticlassConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]]))
        - MulticlassConfusionMatrixTracker(torch.tensor([[1, 0], [1, 2]]))
    ).equal(MulticlassConfusionMatrixTracker(torch.tensor([[2, 2], [0, 2]])))


def test_multiclass_confusion_matrix_add() -> None:
    meter = MulticlassConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).add(
        MulticlassConfusionMatrixTracker(torch.tensor([[1, 0], [7, 2]]))
    )
    assert meter.equal(MulticlassConfusionMatrixTracker(torch.tensor([[4, 2], [8, 6]])))
    assert meter.count == 20


def test_multiclass_confusion_matrix_add_() -> None:
    meter = MulticlassConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]]))
    meter.add_(MulticlassConfusionMatrixTracker(torch.tensor([[1, 0], [7, 2]])))
    assert meter.equal(MulticlassConfusionMatrixTracker(torch.tensor([[4, 2], [8, 6]])))
    assert meter.count == 20


def test_multiclass_confusion_matrix_merge() -> None:
    meter = MulticlassConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]]))
    meter_merged = meter.merge(
        [
            MulticlassConfusionMatrixTracker(torch.tensor([[1, 0], [7, 2]])),
            MulticlassConfusionMatrixTracker(torch.tensor([[1, 0], [0, 1]])),
        ]
    )
    assert meter.equal(MulticlassConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])))
    assert meter.count == 10
    assert meter_merged.equal(MulticlassConfusionMatrixTracker(torch.tensor([[5, 2], [8, 7]])))
    assert meter_merged.count == 22


def test_multiclass_confusion_matrix_merge_() -> None:
    meter = MulticlassConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]]))
    meter.merge_(
        [
            MulticlassConfusionMatrixTracker(torch.tensor([[1, 0], [7, 2]])),
            MulticlassConfusionMatrixTracker(torch.tensor([[1, 0], [0, 1]])),
        ]
    )
    assert meter.equal(MulticlassConfusionMatrixTracker(torch.tensor([[5, 2], [8, 7]])))
    assert meter.count == 22


def test_multiclass_confusion_matrix_sub() -> None:
    meter = MulticlassConfusionMatrixTracker(torch.tensor([[3, 2], [1, 4]])).sub(
        MulticlassConfusionMatrixTracker(torch.tensor([[1, 0], [1, 2]]))
    )
    assert meter.equal(MulticlassConfusionMatrixTracker(torch.tensor([[2, 2], [0, 2]])))
    assert meter.count == 6


# *******************
# *     Metrics     *
# *******************


def test_multiclass_confusion_matrix_false_negative() -> None:
    assert MulticlassConfusionMatrixTracker(
        torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
    ).false_negative.equal(torch.tensor([7, 1, 6], dtype=torch.long))


def test_multiclass_confusion_matrix_false_positive() -> None:
    assert MulticlassConfusionMatrixTracker(
        torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
    ).false_positive.equal(torch.tensor([5, 4, 5], dtype=torch.long))


def test_multiclass_confusion_matrix_support() -> None:
    assert MulticlassConfusionMatrixTracker(
        torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
    ).support.equal(torch.tensor([10, 5, 10], dtype=torch.long))


def test_multiclass_confusion_matrix_true_positive() -> None:
    assert MulticlassConfusionMatrixTracker(
        torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
    ).true_positive.equal(torch.tensor([3, 4, 4], dtype=torch.long))


def test_multiclass_confusion_matrix_accuracy() -> None:
    assert (
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).accuracy()
        == 0.44
    )


def test_multiclass_confusion_matrix_accuracy_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match="It is not possible to compute the accuracy because the confusion matrix is empty",
    ):
        MulticlassConfusionMatrixTracker.from_num_classes(3).accuracy()


def test_multiclass_confusion_matrix_balanced_accuracy() -> None:
    assert (
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).balanced_accuracy()
        == 0.5
    )


def test_multiclass_confusion_matrix_balanced_accuracy_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match=(
            "It is not possible to compute the balanced accuracy because "
            "the confusion matrix is empty"
        ),
    ):
        MulticlassConfusionMatrixTracker.from_num_classes(3).balanced_accuracy()


def test_multiclass_confusion_matrix_f_beta_score_1() -> None:
    assert (
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        )
        .f_beta_score()
        .allclose(
            torch.tensor(
                [0.3333333333333333, 0.6153846153846154, 0.42105263157894735], dtype=torch.float
            )
        )
    )


def test_multiclass_confusion_matrix_f_beta_score_2() -> None:
    assert (
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        )
        .f_beta_score(beta=2)
        .allclose(
            torch.tensor([0.3125, 0.7142857142857143, 0.40816326530612246], dtype=torch.float)
        )
    )


def test_multiclass_confusion_matrix_f_beta_score_0_5() -> None:
    assert (
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        )
        .f_beta_score(beta=0.5)
        .allclose(
            torch.tensor(
                [0.35714285714285715, 0.5405405405405406, 0.43478260869565216], dtype=torch.float
            )
        )
    )


def test_multiclass_confusion_matrix_f_beta_score_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match=(
            "It is not possible to compute the F-beta score because "
            "the confusion matrix is empty"
        ),
    ):
        MulticlassConfusionMatrixTracker.from_num_classes(3).f_beta_score()


def test_multiclass_confusion_matrix_macro_f_beta_score_1() -> None:
    assert math.isclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).macro_f_beta_score(),
        0.4565901756286621,
    )


def test_multiclass_confusion_matrix_macro_f_beta_score_2() -> None:
    assert math.isclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).macro_f_beta_score(beta=2),
        0.4783163368701935,
    )


def test_multiclass_confusion_matrix_macro_f_beta_score_0_5() -> None:
    assert math.isclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).macro_f_beta_score(beta=0.5),
        0.44415533542633057,
    )


def test_multiclass_confusion_matrix_macro_f_beta_score_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match=(
            "It is not possible to compute the F-beta score because "
            "the confusion matrix is empty"
        ),
    ):
        MulticlassConfusionMatrixTracker.from_num_classes(3).macro_f_beta_score()


def test_multiclass_confusion_matrix_micro_f_beta_score_1() -> None:
    assert math.isclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).micro_f_beta_score(),
        0.44,
        abs_tol=1e-6,
    )


def test_multiclass_confusion_matrix_micro_f_beta_score_2() -> None:
    assert math.isclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).micro_f_beta_score(beta=2),
        0.44,
        abs_tol=1e-6,
    )


def test_multiclass_confusion_matrix_micro_f_beta_score_0_5() -> None:
    assert math.isclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).micro_f_beta_score(beta=0.5),
        0.44,
        abs_tol=1e-6,
    )


def test_multiclass_confusion_matrix_micro_f_beta_score_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match=(
            "It is not possible to compute the micro F-beta score because "
            "the confusion matrix is empty"
        ),
    ):
        MulticlassConfusionMatrixTracker.from_num_classes(3).micro_f_beta_score()


def test_multiclass_confusion_matrix_weighted_f_beta_score_1() -> None:
    assert math.isclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).weighted_f_beta_score(),
        0.42483131408691405,
        abs_tol=1e-6,
    )


def test_multiclass_confusion_matrix_weighted_f_beta_score_2() -> None:
    assert math.isclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).weighted_f_beta_score(beta=2),
        0.4311224365234375,
        abs_tol=1e-6,
    )


def test_multiclass_confusion_matrix_weighted_f_beta_score_0_5() -> None:
    assert math.isclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).weighted_f_beta_score(beta=0.5),
        0.4248783111572266,
        abs_tol=1e-6,
    )


def test_multiclass_confusion_matrix_weighted_f_beta_score_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match=(
            "It is not possible to compute the F-beta score because "
            "the confusion matrix is empty"
        ),
    ):
        MulticlassConfusionMatrixTracker.from_num_classes(3).weighted_f_beta_score()


def test_multiclass_confusion_matrix_precision() -> None:
    assert (
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 1], [4, 2, 4]], dtype=torch.long)
        )
        .precision()
        .equal(torch.tensor([0.375, 0.5, 0.4], dtype=torch.float))
    )


def test_multiclass_confusion_matrix_precision_zero() -> None:
    assert (
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 0, 5], [3, 0, 1], [4, 0, 4]], dtype=torch.long)
        )
        .precision()
        .equal(torch.tensor([0.3, 0.0, 0.4], dtype=torch.float))
    )


def test_multiclass_confusion_matrix_precision_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match="It is not possible to compute the precision because the confusion matrix is empty",
    ):
        MulticlassConfusionMatrixTracker.from_num_classes(3).precision()


def test_multiclass_confusion_matrix_macro_precision() -> None:
    assert math.isclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).macro_precision(),
        0.43981480598449707,
        abs_tol=1e-6,
    )


def test_multiclass_confusion_matrix_macro_precision_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match="It is not possible to compute the precision because the confusion matrix is empty",
    ):
        MulticlassConfusionMatrixTracker.from_num_classes(3).macro_precision()


def test_multiclass_confusion_matrix_micro_precision() -> None:
    assert math.isclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).micro_precision(),
        0.44,
        abs_tol=1e-6,
    )


def test_multiclass_confusion_matrix_micro_precision_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match=(
            "It is not possible to compute the micro precision because "
            "the confusion matrix is empty"
        ),
    ):
        MulticlassConfusionMatrixTracker.from_num_classes(3).micro_precision()


def test_multiclass_confusion_matrix_weighted_precision() -> None:
    assert (
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).weighted_precision()
        == 0.42777778625488283
    )


def test_multiclass_confusion_matrix_weighted_precision_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match=("It is not possible to compute the precision because the confusion matrix is empty"),
    ):
        MulticlassConfusionMatrixTracker.from_num_classes(3).weighted_precision()


def test_multiclass_confusion_matrix_recall() -> None:
    assert (
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        )
        .recall()
        .equal(torch.tensor([0.3, 0.8, 0.4], dtype=torch.float))
    )


def test_multiclass_confusion_matrix_recall_zero() -> None:
    assert (
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [0, 0, 0], [4, 2, 4]], dtype=torch.long)
        )
        .recall()
        .equal(torch.tensor([0.3, 0.0, 0.4], dtype=torch.float))
    )


def test_multiclass_confusion_matrix_recall_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match="It is not possible to compute the recall because the confusion matrix is empty",
    ):
        MulticlassConfusionMatrixTracker.from_num_classes(3).recall()


def test_multiclass_confusion_matrix_macro_recall() -> None:
    assert (
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).macro_recall()
        == 0.5
    )


def test_multiclass_confusion_matrix_macro_recall_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match="It is not possible to compute the recall because the confusion matrix is empty",
    ):
        MulticlassConfusionMatrixTracker.from_num_classes(3).macro_recall()


def test_multiclass_confusion_matrix_micro_recall() -> None:
    assert math.isclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).micro_recall(),
        0.44,
        abs_tol=1e-6,
    )


def test_multiclass_confusion_matrix_micro_recall_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match=(
            "It is not possible to compute the micro recall because "
            "the confusion matrix is empty"
        ),
    ):
        MulticlassConfusionMatrixTracker.from_num_classes(3).micro_recall()


def test_multiclass_confusion_matrix_weighted_recall() -> None:
    assert math.isclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).weighted_recall(),
        0.44,
        abs_tol=1e-6,
    )


def test_multiclass_confusion_matrix_weighted_recall_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match="It is not possible to compute the recall because the confusion matrix is empty",
    ):
        MulticlassConfusionMatrixTracker.from_num_classes(3).weighted_recall()


def test_multiclass_confusion_matrix_compute_per_class_metrics() -> None:
    assert objects_are_allclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_per_class_metrics(),
        {
            "f1_score": torch.tensor([0.3333333432674408, 0.6153846383094788, 0.42105263471603394]),
            "precision": torch.tensor([0.375, 0.5, 0.4444444477558136]),
            "recall": torch.tensor([0.3, 0.8, 0.4]),
        },
    )


def test_multiclass_confusion_matrix_compute_per_class_metrics_betas() -> None:
    assert objects_are_allclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_per_class_metrics(betas=(1, 2)),
        {
            "f1_score": torch.tensor([0.3333333432674408, 0.6153846383094788, 0.42105263471603394]),
            "f2_score": torch.tensor([0.3125, 0.7142857313156128, 0.40816327929496765]),
            "precision": torch.tensor([0.375, 0.5, 0.4444444477558136]),
            "recall": torch.tensor([0.3, 0.8, 0.4]),
        },
    )


def test_multiclass_confusion_matrix_compute_per_class_metrics_prefix_suffix() -> None:
    assert objects_are_allclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_per_class_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_f1_score_suffix": torch.tensor(
                [0.3333333432674408, 0.6153846383094788, 0.42105263471603394]
            ),
            "prefix_precision_suffix": torch.tensor([0.375, 0.5, 0.4444444477558136]),
            "prefix_recall_suffix": torch.tensor([0.3, 0.8, 0.4]),
        },
    )


def test_multiclass_confusion_matrix_compute_per_class_metrics_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match="It is not possible to compute the metrics because the confusion matrix is empty",
    ):
        MulticlassConfusionMatrixTracker.from_num_classes(3).compute_per_class_metrics()


def test_multiclass_confusion_matrix_compute_macro_metrics() -> None:
    assert objects_are_allclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_macro_metrics(),
        {
            "macro_f1_score": 0.4565901756286621,
            "macro_precision": 0.43981480598449707,
            "macro_recall": 0.5,
        },
    )


def test_multiclass_confusion_matrix_compute_macro_metrics_betas() -> None:
    assert objects_are_allclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_macro_metrics(betas=(1, 2)),
        {
            "macro_f1_score": 0.4565901756286621,
            "macro_f2_score": 0.4783163368701935,
            "macro_precision": 0.43981480598449707,
            "macro_recall": 0.5,
        },
    )


def test_multiclass_confusion_matrix_compute_macro_metrics_prefix_suffix() -> None:
    assert objects_are_allclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_macro_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_macro_f1_score_suffix": 0.4565901756286621,
            "prefix_macro_precision_suffix": 0.43981480598449707,
            "prefix_macro_recall_suffix": 0.5,
        },
    )


def test_multiclass_confusion_matrix_compute_macro_metrics_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match=(
            "It is not possible to compute the 'macro' metrics because "
            "the confusion matrix is empty"
        ),
    ):
        MulticlassConfusionMatrixTracker.from_num_classes(3).compute_macro_metrics()


def test_multiclass_confusion_matrix_compute_micro_metrics() -> None:
    assert objects_are_allclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_micro_metrics(),
        {
            "micro_f1_score": 0.44,
            "micro_precision": 0.44,
            "micro_recall": 0.44,
        },
    )


def test_multiclass_confusion_matrix_compute_micro_metrics_betas() -> None:
    assert objects_are_allclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_micro_metrics(betas=(1, 2)),
        {
            "micro_f1_score": 0.44,
            "micro_f2_score": 0.44,
            "micro_precision": 0.44,
            "micro_recall": 0.44,
        },
    )


def test_multiclass_confusion_matrix_compute_micro_metrics_prefix_suffix() -> None:
    assert objects_are_allclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_micro_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_micro_f1_score_suffix": 0.44,
            "prefix_micro_precision_suffix": 0.44,
            "prefix_micro_recall_suffix": 0.44,
        },
    )


def test_multiclass_confusion_matrix_compute_micro_metrics_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match=(
            "It is not possible to compute the 'micro' metrics because "
            "the confusion matrix is empty"
        ),
    ):
        MulticlassConfusionMatrixTracker.from_num_classes(3).compute_micro_metrics()


def test_multiclass_confusion_matrix_compute_weighted_metrics() -> None:
    assert objects_are_allclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_weighted_metrics(),
        {
            "weighted_f1_score": 0.42483131408691405,
            "weighted_precision": 0.42777778625488283,
            "weighted_recall": 0.44,
        },
    )


def test_multiclass_confusion_matrix_compute_weighted_metrics_betas() -> None:
    assert objects_are_allclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_weighted_metrics(betas=(1, 2)),
        {
            "weighted_f1_score": 0.42483131408691405,
            "weighted_f2_score": 0.4311224365234375,
            "weighted_precision": 0.42777778625488283,
            "weighted_recall": 0.44,
        },
    )


def test_multiclass_confusion_matrix_compute_weighted_metrics_prefix_suffix() -> None:
    assert objects_are_allclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_weighted_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_weighted_f1_score_suffix": 0.42483131408691405,
            "prefix_weighted_precision_suffix": 0.42777778625488283,
            "prefix_weighted_recall_suffix": 0.44,
        },
    )


def test_multiclass_confusion_matrix_compute_weighted_metrics_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match=(
            "It is not possible to compute the 'weighted' metrics because "
            "the confusion matrix is empty"
        ),
    ):
        MulticlassConfusionMatrixTracker.from_num_classes(3).compute_weighted_metrics()


def test_multiclass_confusion_matrix_compute_scalar_metrics() -> None:
    assert objects_are_allclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_scalar_metrics(),
        {
            "accuracy": 0.44,
            "balanced_accuracy": 0.5,
            "macro_f1_score": 0.4565901756286621,
            "macro_precision": 0.43981480598449707,
            "macro_recall": 0.5,
            "micro_f1_score": 0.44,
            "micro_precision": 0.44,
            "micro_recall": 0.44,
            "weighted_f1_score": 0.42483131408691405,
            "weighted_precision": 0.42777778625488283,
            "weighted_recall": 0.44,
        },
    )


def test_multiclass_confusion_matrix_compute_scalar_metrics_betas() -> None:
    assert objects_are_allclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_scalar_metrics(betas=(1, 2)),
        {
            "accuracy": 0.44,
            "balanced_accuracy": 0.5,
            "macro_f1_score": 0.4565901756286621,
            "macro_f2_score": 0.4783163368701935,
            "macro_precision": 0.43981480598449707,
            "macro_recall": 0.5,
            "micro_f1_score": 0.44,
            "micro_f2_score": 0.44,
            "micro_precision": 0.44,
            "micro_recall": 0.44,
            "weighted_f1_score": 0.42483131408691405,
            "weighted_f2_score": 0.4311224365234375,
            "weighted_precision": 0.42777778625488283,
            "weighted_recall": 0.44,
        },
    )


def test_multiclass_confusion_matrix_compute_scalar_metrics_prefix_suffix() -> None:
    assert objects_are_allclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_scalar_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_accuracy_suffix": 0.44,
            "prefix_balanced_accuracy_suffix": 0.5,
            "prefix_macro_f1_score_suffix": 0.4565901756286621,
            "prefix_macro_precision_suffix": 0.43981480598449707,
            "prefix_macro_recall_suffix": 0.5,
            "prefix_micro_f1_score_suffix": 0.44,
            "prefix_micro_precision_suffix": 0.44,
            "prefix_micro_recall_suffix": 0.44,
            "prefix_weighted_f1_score_suffix": 0.42483131408691405,
            "prefix_weighted_precision_suffix": 0.42777778625488283,
            "prefix_weighted_recall_suffix": 0.44,
        },
    )


def test_multiclass_confusion_matrix_compute_scalar_metrics_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match="It is not possible to compute the metrics because the confusion matrix is empty",
    ):
        MulticlassConfusionMatrixTracker.from_num_classes(3).compute_scalar_metrics()


def test_multiclass_confusion_matrix_compute_metrics() -> None:
    assert objects_are_allclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_metrics(),
        {
            "accuracy": 0.44,
            "balanced_accuracy": 0.5,
            "macro_f1_score": 0.4565901756286621,
            "macro_precision": 0.43981480598449707,
            "macro_recall": 0.5,
            "micro_f1_score": 0.44,
            "micro_precision": 0.44,
            "micro_recall": 0.44,
            "weighted_f1_score": 0.42483131408691405,
            "weighted_precision": 0.42777778625488283,
            "weighted_recall": 0.44,
            "precision": torch.tensor([0.375, 0.5, 0.4444444477558136]),
            "recall": torch.tensor([0.30000001192092896, 0.800000011920929, 0.4000000059604645]),
            "f1_score": torch.tensor([0.3333333432674408, 0.6153846383094788, 0.42105263471603394]),
        },
    )


def test_multiclass_confusion_matrix_compute_metrics_betas() -> None:
    assert objects_are_allclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_metrics(betas=(1, 2)),
        {
            "accuracy": 0.44,
            "balanced_accuracy": 0.5,
            "macro_f1_score": 0.4565901756286621,
            "macro_f2_score": 0.4783163368701935,
            "macro_precision": 0.43981480598449707,
            "macro_recall": 0.5,
            "micro_f1_score": 0.44,
            "micro_f2_score": 0.44,
            "micro_precision": 0.44,
            "micro_recall": 0.44,
            "weighted_f1_score": 0.42483131408691405,
            "weighted_f2_score": 0.4311224365234375,
            "weighted_precision": 0.42777778625488283,
            "weighted_recall": 0.44,
            "precision": torch.tensor([0.375, 0.5, 0.4444444477558136]),
            "recall": torch.tensor([0.30000001192092896, 0.800000011920929, 0.4000000059604645]),
            "f1_score": torch.tensor([0.3333333432674408, 0.6153846383094788, 0.42105263471603394]),
            "f2_score": torch.tensor([0.3125, 0.7142857313156128, 0.40816327929496765]),
        },
    )


def test_multiclass_confusion_matrix_compute_metrics_prefix_suffix() -> None:
    assert objects_are_allclose(
        MulticlassConfusionMatrixTracker(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_accuracy_suffix": 0.44,
            "prefix_balanced_accuracy_suffix": 0.5,
            "prefix_macro_f1_score_suffix": 0.4565901756286621,
            "prefix_macro_precision_suffix": 0.43981480598449707,
            "prefix_macro_recall_suffix": 0.5,
            "prefix_micro_f1_score_suffix": 0.44,
            "prefix_micro_precision_suffix": 0.44,
            "prefix_micro_recall_suffix": 0.44,
            "prefix_weighted_f1_score_suffix": 0.42483131408691405,
            "prefix_weighted_precision_suffix": 0.42777778625488283,
            "prefix_weighted_recall_suffix": 0.44,
            "prefix_precision_suffix": torch.tensor([0.375, 0.5, 0.4444444477558136]),
            "prefix_recall_suffix": torch.tensor(
                [0.30000001192092896, 0.800000011920929, 0.4000000059604645]
            ),
            "prefix_f1_score_suffix": torch.tensor(
                [0.3333333432674408, 0.6153846383094788, 0.42105263471603394]
            ),
        },
    )


def test_multiclass_confusion_matrix_compute_metrics_empty() -> None:
    with pytest.raises(
        EmptyTrackerError,
        match="It is not possible to compute the metrics because the confusion matrix is empty",
    ):
        MulticlassConfusionMatrixTracker.from_num_classes(3).compute_metrics()


def test_multiclass_confusion_matrix_get_records() -> None:
    assert objects_are_allclose(
        MulticlassConfusionMatrixTracker.from_num_classes(5).get_records(),
        (
            MaxScalarRecord(name="accuracy"),
            MaxScalarRecord(name="balanced_accuracy"),
            MaxScalarRecord(name="macro_precision"),
            MaxScalarRecord(name="macro_recall"),
            MaxScalarRecord(name="micro_precision"),
            MaxScalarRecord(name="micro_recall"),
            MaxScalarRecord(name="weighted_precision"),
            MaxScalarRecord(name="weighted_recall"),
            MaxScalarRecord(name="macro_f1_score"),
            MaxScalarRecord(name="micro_f1_score"),
            MaxScalarRecord(name="weighted_f1_score"),
        ),
    )


def test_multiclass_confusion_matrix_get_records_betas() -> None:
    assert objects_are_allclose(
        MulticlassConfusionMatrixTracker.from_num_classes(5).get_records(betas=[0.5, 1, 2]),
        (
            MaxScalarRecord(name="accuracy"),
            MaxScalarRecord(name="balanced_accuracy"),
            MaxScalarRecord(name="macro_precision"),
            MaxScalarRecord(name="macro_recall"),
            MaxScalarRecord(name="micro_precision"),
            MaxScalarRecord(name="micro_recall"),
            MaxScalarRecord(name="weighted_precision"),
            MaxScalarRecord(name="weighted_recall"),
            MaxScalarRecord(name="macro_f0.5_score"),
            MaxScalarRecord(name="micro_f0.5_score"),
            MaxScalarRecord(name="weighted_f0.5_score"),
            MaxScalarRecord(name="macro_f1_score"),
            MaxScalarRecord(name="micro_f1_score"),
            MaxScalarRecord(name="weighted_f1_score"),
            MaxScalarRecord(name="macro_f2_score"),
            MaxScalarRecord(name="micro_f2_score"),
            MaxScalarRecord(name="weighted_f2_score"),
        ),
    )


def test_multiclass_confusion_matrix_get_records_prefix_suffix() -> None:
    assert objects_are_allclose(
        MulticlassConfusionMatrixTracker.from_num_classes(5).get_records(
            prefix="prefix_", suffix="_suffix"
        ),
        (
            MaxScalarRecord(name="prefix_accuracy_suffix"),
            MaxScalarRecord(name="prefix_balanced_accuracy_suffix"),
            MaxScalarRecord(name="prefix_macro_precision_suffix"),
            MaxScalarRecord(name="prefix_macro_recall_suffix"),
            MaxScalarRecord(name="prefix_micro_precision_suffix"),
            MaxScalarRecord(name="prefix_micro_recall_suffix"),
            MaxScalarRecord(name="prefix_weighted_precision_suffix"),
            MaxScalarRecord(name="prefix_weighted_recall_suffix"),
            MaxScalarRecord(name="prefix_macro_f1_score_suffix"),
            MaxScalarRecord(name="prefix_micro_f1_score_suffix"),
            MaxScalarRecord(name="prefix_weighted_f1_score_suffix"),
        ),
    )


############################################
#     Tests for check_confusion_matrix     #
############################################


def test_check_confusion_matrix_incorrect_ndim() -> None:
    with pytest.raises(ValueError, match="Incorrect matrix dimensions."):
        check_confusion_matrix(torch.zeros(3))


def test_check_confusion_matrix_incorrect_shape() -> None:
    with pytest.raises(ValueError, match="Incorrect matrix shape."):
        check_confusion_matrix(torch.zeros(3, 5))


def test_check_confusion_matrix_incorrect_dtype() -> None:
    with pytest.raises(ValueError, match="Incorrect matrix data type."):
        check_confusion_matrix(torch.zeros(2, 2, dtype=torch.float))


def test_check_confusion_matrix_negative_value() -> None:
    with pytest.raises(ValueError, match="Incorrect matrix values."):
        check_confusion_matrix(torch.tensor([[0, 0], [-1, 0]]))


###################################################
#     Tests for check_op_compatibility_binary     #
###################################################


def test_check_op_compatibility_binary_correct() -> None:
    check_op_compatibility_binary(
        BinaryConfusionMatrixTracker(), BinaryConfusionMatrixTracker(), "op"
    )
    # will fail if an exception is raised


def test_check_op_compatibility_binary_incorrect_type() -> None:
    with pytest.raises(TypeError, match="Incorrect type .*Mock.*"):
        check_op_compatibility_binary(BinaryConfusionMatrixTracker(), Mock(), "op")


#######################################################
#     Tests for check_op_compatibility_multiclass     #
#######################################################


def test_check_op_compatibility_multiclass_correct() -> None:
    check_op_compatibility_multiclass(
        MulticlassConfusionMatrixTracker.from_num_classes(3),
        MulticlassConfusionMatrixTracker.from_num_classes(3),
        "op",
    )
    # will fail if an exception is raised


def test_check_op_compatibility_multiclass_incorrect_type() -> None:
    with pytest.raises(TypeError, match="Incorrect type: .*Mock.*"):
        check_op_compatibility_multiclass(
            MulticlassConfusionMatrixTracker.from_num_classes(3),
            Mock(),
            "op",
        )


def test_check_op_compatibility_multiclass_incorrect_shape() -> None:
    with pytest.raises(ValueError, match="Incorrect shape:"):
        check_op_compatibility_multiclass(
            MulticlassConfusionMatrixTracker.from_num_classes(3),
            MulticlassConfusionMatrixTracker.from_num_classes(4),
            "op",
        )


#######################################
#     str_binary_confusion_matrix     #
#######################################


def test_str_binary_confusion_matrix() -> None:
    assert isinstance(str_binary_confusion_matrix(torch.tensor([[1001, 42], [123, 789]])), str)


@tabulate_available
def test_str_binary_confusion_matrix_tabulate() -> None:
    assert str_binary_confusion_matrix(torch.tensor([[1001, 42], [123, 789]])) == (
        "┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
        "┃                     ┃ predicted negative (0) ┃ predicted positive (1) ┃\n"
        "┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫\n"
        "┃ actual negative (0) ┃ [TN]  1,001            ┃ [FP]  42               ┃\n"
        "┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫\n"
        "┃ actual positive (1) ┃ [FN]  123              ┃ [TP]  789              ┃\n"
        "┗━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┛"
    )


def test_str_binary_confusion_matrix_empty() -> None:
    assert isinstance(str_binary_confusion_matrix(torch.zeros(2, 2)), str)


@tabulate_available
def test_str_binary_confusion_matrix_empty_tabulate() -> None:
    assert str_binary_confusion_matrix(torch.zeros(2, 2)) == (
        "┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
        "┃                     ┃ predicted negative (0) ┃ predicted positive (1) ┃\n"
        "┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫\n"
        "┃ actual negative (0) ┃ [TN]  0                ┃ [FP]  0                ┃\n"
        "┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫\n"
        "┃ actual positive (1) ┃ [FN]  0                ┃ [TP]  0                ┃\n"
        "┗━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┛"
    )


@pytest.mark.parametrize("shape", [(1,), (1, 1), (2, 3), (3, 2), (3, 3)])
def test_str_binary_confusion_matrix_incorrect_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(RuntimeError, match="Expected a 2x2 confusion matrix but received"):
        str_binary_confusion_matrix(torch.zeros(*shape))
