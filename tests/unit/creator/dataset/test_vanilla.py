from __future__ import annotations

from karbonn.creator.dataset import DatasetCreator
from karbonn.testing import objectory_available
from karbonn.testing.dummy import DummyDataset
from karbonn.utils.imports import is_objectory_available

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"

####################################
#     Tests for DatasetCreator     #
####################################


def test_dataset_creator_repr() -> None:
    assert repr(
        DatasetCreator(
            dataset={
                OBJECT_TARGET: "karbonn.testing.dummy.DummyDataset",
                "num_examples": 10,
                "feature_size": 4,
            }
        )
    ).startswith("DatasetCreator")


def test_dataset_creator_str() -> None:
    assert str(
        DatasetCreator(
            dataset={
                OBJECT_TARGET: "karbonn.testing.dummy.DummyDataset",
                "num_examples": 10,
                "feature_size": 4,
            }
        )
    ).startswith("DatasetCreator")


@objectory_available
def test_dataset_creator_create_dict() -> None:
    assert isinstance(
        DatasetCreator(
            dataset={
                OBJECT_TARGET: "karbonn.testing.dummy.DummyDataset",
                "num_examples": 10,
                "feature_size": 4,
            }
        ).create(),
        DummyDataset,
    )


def test_dataset_creator_create_object() -> None:
    dataset = DummyDataset()
    assert DatasetCreator(dataset=dataset).create() is dataset
