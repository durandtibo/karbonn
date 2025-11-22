from __future__ import annotations

import logging
from collections import defaultdict
from unittest.mock import patch

import pytest

from karbonn.creator.optimizer import (
    OptimizerCreator,
    is_optimizer_creator_config,
    setup_optimizer_creator,
)
from karbonn.testing import objectory_available
from karbonn.utils.imports import is_objectory_available

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"


#################################################
#     Tests for is_optimizer_creator_config     #
#################################################


@objectory_available
def test_is_optimizer_creator_config_true() -> None:
    assert is_optimizer_creator_config(
        {
            OBJECT_TARGET: "karbonn.creator.optimizer.OptimizerCreator",
            "optimizer": {OBJECT_TARGET: "torch.optim.SGD", "lr": 0.001},
        }
    )


@objectory_available
def test_is_optimizer_creator_config_false() -> None:
    assert not is_optimizer_creator_config({OBJECT_TARGET: "torch.nn.Identity"})


#############################################
#     Tests for setup_optimizer_creator     #
#############################################


def test_setup_optimizer_creator_object() -> None:
    creator = OptimizerCreator(optimizer={OBJECT_TARGET: "torch.optim.SGD", "lr": 0.001})
    assert setup_optimizer_creator(creator) is creator


@objectory_available
def test_setup_optimizer_creator_config() -> None:
    assert isinstance(
        setup_optimizer_creator(
            {
                OBJECT_TARGET: "karbonn.creator.optimizer.OptimizerCreator",
                "optimizer": {OBJECT_TARGET: "torch.optim.SGD", "lr": 0.001},
            }
        ),
        OptimizerCreator,
    )


@objectory_available
def test_setup_optimizer_creator_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_optimizer_creator({OBJECT_TARGET: "collections.defaultdict"}), defaultdict
        )
        assert caplog.messages


def test_setup_optimizer_creator_object_no_objectory() -> None:
    with (
        patch("karbonn.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."),
    ):
        setup_optimizer_creator(
            {
                OBJECT_TARGET: "karbonn.creator.optimizer.OptimizerCreator",
                "optimizer": {OBJECT_TARGET: "torch.optim.SGD", "lr": 0.001},
            }
        )
