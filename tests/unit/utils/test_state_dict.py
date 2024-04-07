from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from karbonn.utils import find_module_state_dict

LINEAR_STATE_DICT = {"weight": torch.ones(5, 4), "bias": 2 * torch.ones(5)}

STATE_DICTS = [
    {"model": {"network": LINEAR_STATE_DICT}},
    {"list": ["weight", "bias"], "model": {"network": LINEAR_STATE_DICT}},  # should not be detected
    {"set": {"weight", "bias"}, "model": {"network": LINEAR_STATE_DICT}},  # should not be detected
    {
        "tuple": ("weight", "bias"),
        "model": {"network": LINEAR_STATE_DICT},
    },  # should not be detected
    {"list": ["weight", "bias", LINEAR_STATE_DICT], "abc": None},
]


############################################
#     Tests for find_module_state_dict     #
############################################


def test_find_module_state_dict() -> None:
    state_dict = {"weight": torch.ones(5, 4), "bias": 2 * torch.ones(5)}
    assert objects_are_equal(state_dict, find_module_state_dict(state_dict, {"weight", "bias"}))


@pytest.mark.parametrize("state_dict", STATE_DICTS)
def test_find_module_state_dict_nested(state_dict: dict) -> None:
    assert objects_are_equal(
        LINEAR_STATE_DICT, find_module_state_dict(state_dict, {"bias", "weight"})
    )


def test_find_module_state_dict_missing_key() -> None:
    assert find_module_state_dict({"weight": torch.ones(5, 4)}, {"bias", "weight"}) == {}
