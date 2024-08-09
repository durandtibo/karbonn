from __future__ import annotations

import logging
from collections import defaultdict
from unittest.mock import patch

import pytest

from karbonn.metric import AbsoluteError, setup_metric
from karbonn.testing import objectory_available
from karbonn.utils.imports import is_objectory_available

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"


##################################
#     Tests for setup_metric     #
##################################


def test_setup_metric_object() -> None:
    metric = AbsoluteError()
    assert setup_metric(metric) is metric


@objectory_available
def test_setup_metric_config() -> None:
    assert isinstance(
        setup_metric({OBJECT_TARGET: "karbonn.metric.AbsoluteError"}),
        AbsoluteError,
    )


@objectory_available
def test_setup_metric_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_metric({OBJECT_TARGET: "collections.defaultdict"}), defaultdict)
        assert caplog.messages


def test_setup_metric_object_no_objectory() -> None:
    with (
        patch("karbonn.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match="'objectory' package is required but not installed."),
    ):
        setup_metric({OBJECT_TARGET: "karbonn.metric.AbsoluteError"})
