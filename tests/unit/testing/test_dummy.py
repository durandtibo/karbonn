from __future__ import annotations

import pytest

from karbonn.testing.dummy import DummyDataset

##################################
#     Tests for DummyDataset     #
##################################


def test_dummy_dataset_repr() -> None:
    assert repr(DummyDataset()).startswith("DummyDataset(")


def test_dummy_dataset_str() -> None:
    assert str(DummyDataset()).startswith("DummyDataset(")


@pytest.mark.parametrize("num_examples", [1, 3, 5])
def test_dummy_dataset_num_examples(num_examples: int) -> None:
    dataset = DummyDataset(num_examples=num_examples)
    assert len(dataset) == num_examples
    assert dataset._features.shape == (num_examples, 4)
    assert dataset._target.shape == (num_examples, 1)


@pytest.mark.parametrize("feature_size", [1, 3, 5])
def test_dummy_dataset_feature_size(feature_size: int) -> None:
    dataset = DummyDataset(feature_size=feature_size)
    assert dataset._features.shape == (8, feature_size)
    assert dataset._target.shape == (8, 1)


def test_dummy_dataset_feature_getitem() -> None:
    sample = DummyDataset()[0]
    assert isinstance(sample, dict)
    assert sample["feature"].shape == (4,)
    assert sample["target"].shape == (1,)
