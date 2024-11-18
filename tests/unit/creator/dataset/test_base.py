from __future__ import annotations

import logging
from collections import defaultdict
from unittest.mock import patch

import pytest

from karbonn.creator.dataset import (
    DatasetCreator,
    is_dataset_creator_config,
    setup_dataset_creator,
)
from karbonn.testing import objectory_available
from karbonn.testing.dummy import DummyDataset
from karbonn.utils.imports import is_objectory_available

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"


###############################################
#     Tests for is_dataset_creator_config     #
###############################################


@objectory_available
def test_is_dataset_creator_config_true() -> None:
    assert is_dataset_creator_config(
        {
            OBJECT_TARGET: "karbonn.creator.dataset.DatasetCreator",
            "dataset": {
                OBJECT_TARGET: "karbonn.testing.dummy.DummyDataset",
                "num_examples": 10,
                "feature_size": 4,
            },
        }
    )


@objectory_available
def test_is_dataset_creator_config_false() -> None:
    assert not is_dataset_creator_config({OBJECT_TARGET: "torch.nn.Identity"})


###########################################
#     Tests for setup_dataset_creator     #
###########################################


def test_setup_dataset_creator_object() -> None:
    creator = DatasetCreator(DummyDataset())
    assert setup_dataset_creator(creator) is creator


@objectory_available
def test_setup_dataset_creator_config() -> None:
    assert isinstance(
        setup_dataset_creator(
            {
                OBJECT_TARGET: "karbonn.creator.dataset.DatasetCreator",
                "dataset": {
                    OBJECT_TARGET: "karbonn.testing.dummy.DummyDataset",
                    "num_examples": 10,
                    "feature_size": 4,
                },
            }
        ),
        DatasetCreator,
    )


@objectory_available
def test_setup_dataset_creator_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_dataset_creator({OBJECT_TARGET: "collections.defaultdict"}), defaultdict
        )
        assert caplog.messages


def test_setup_dataset_creator_object_no_objectory() -> None:
    with (
        patch("karbonn.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match="'objectory' package is required but not installed."),
    ):
        setup_dataset_creator(
            {
                OBJECT_TARGET: "karbonn.creator.dataset.DatasetCreator",
                "dataset": {
                    OBJECT_TARGET: "karbonn.testing.dummy.DummyDataset",
                    "num_examples": 10,
                    "feature_size": 4,
                },
            }
        )
