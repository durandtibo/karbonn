from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from objectory import OBJECT_TARGET

from karbonn.metric import AbsoluteError, setup_metric

if TYPE_CHECKING:
    import pytest

##################################
#     Tests for setup_metric     #
##################################


def test_setup_metric_object() -> None:
    metric = AbsoluteError()
    assert setup_metric(metric) is metric


def test_setup_metric_config() -> None:
    assert isinstance(
        setup_metric({OBJECT_TARGET: "karbonn.metric.AbsoluteError"}),
        AbsoluteError,
    )


def test_setup_metric_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_metric({OBJECT_TARGET: "collections.defaultdict"}), defaultdict)
        assert caplog.messages
