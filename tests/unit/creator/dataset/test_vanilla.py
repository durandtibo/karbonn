from __future__ import annotations

from karbonn.creator.dataset import DatasetCreator
from karbonn.testing.dummy import DummyDataset

####################################
#     Tests for DatasetCreator     #
####################################


def test_dataset_creator_repr() -> None:
    assert repr(
        DatasetCreator(
            dataset={
                "_target_": "karbonn.testing.dummy.DummyDataset",
                "num_examples": 10,
                "feature_size": 4,
            }
        )
    ).startswith("DatasetCreator")


def test_dataset_creator_str() -> None:
    assert str(
        DatasetCreator(
            dataset={
                "_target_": "karbonn.testing.dummy.DummyDataset",
                "num_examples": 10,
                "feature_size": 4,
            }
        )
    ).startswith("DatasetCreator")


def test_dataset_creator_create_dict() -> None:
    assert isinstance(
        DatasetCreator(
            dataset={
                "_target_": "karbonn.testing.dummy.DummyDataset",
                "num_examples": 10,
                "feature_size": 4,
            }
        ).create(),
        DummyDataset,
    )


def test_dataset_creator_create_object() -> None:
    dataset = DummyDataset()
    assert DatasetCreator(dataset=dataset).create() is dataset
