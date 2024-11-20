from __future__ import annotations

import logging
from collections import defaultdict
from unittest.mock import patch

import pytest
from torch import nn

from karbonn.creator.module import (
    ModuleCreator,
    is_module_creator_config,
    setup_module_creator,
)
from karbonn.testing import objectory_available
from karbonn.utils.imports import is_objectory_available

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"


###############################################
#     Tests for is_module_creator_config     #
###############################################


@objectory_available
def test_is_module_creator_config_true() -> None:
    assert is_module_creator_config(
        {
            OBJECT_TARGET: "karbonn.creator.module.ModuleCreator",
            "module": {
                OBJECT_TARGET: "torch.nn.Linear",
                "in_features": 4,
                "out_features": 6,
            },
        }
    )


@objectory_available
def test_is_module_creator_config_false() -> None:
    assert not is_module_creator_config({OBJECT_TARGET: "torch.nn.Identity"})


###########################################
#     Tests for setup_module_creator     #
###########################################


def test_setup_module_creator_object() -> None:
    creator = ModuleCreator(nn.Linear(in_features=4, out_features=6))
    assert setup_module_creator(creator) is creator


@objectory_available
def test_setup_module_creator_config() -> None:
    assert isinstance(
        setup_module_creator(
            {
                OBJECT_TARGET: "karbonn.creator.module.ModuleCreator",
                "module": {
                    OBJECT_TARGET: "torch.nn.Linear",
                    "in_features": 4,
                    "out_features": 6,
                },
            }
        ),
        ModuleCreator,
    )


@objectory_available
def test_setup_module_creator_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_module_creator({OBJECT_TARGET: "collections.defaultdict"}), defaultdict
        )
        assert caplog.messages


def test_setup_module_creator_object_no_objectory() -> None:
    with (
        patch("karbonn.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match="'objectory' package is required but not installed."),
    ):
        setup_module_creator(
            {
                OBJECT_TARGET: "karbonn.creator.module.ModuleCreator",
                "module": {
                    OBJECT_TARGET: "torch.nn.Linear",
                    "in_features": 4,
                    "out_features": 6,
                },
            }
        )
