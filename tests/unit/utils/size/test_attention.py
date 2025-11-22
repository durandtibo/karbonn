from __future__ import annotations

import pytest
from torch import nn

from karbonn.utils.size import MultiheadAttentionSizeFinder
from karbonn.utils.size.base import SizeNotFoundError
from tests.unit.utils.size.utils import ModuleSizes

MULTIHEAD_ATTENTION_MODULES = [
    ModuleSizes(
        module=nn.MultiheadAttention(embed_dim=4, num_heads=1),
        in_features=[4, 4, 4],
        out_features=[4],
    ),
    ModuleSizes(
        module=nn.MultiheadAttention(embed_dim=8, num_heads=2),
        in_features=[8, 8, 8],
        out_features=[8],
    ),
    ModuleSizes(
        module=nn.MultiheadAttention(embed_dim=4, num_heads=1, kdim=2, vdim=3),
        in_features=[4, 2, 3],
        out_features=[4],
    ),
]


##################################################
#     Tests for MultiheadAttentionSizeFinder     #
##################################################


def test_multihead_attention_size_finder_repr() -> None:
    assert repr(MultiheadAttentionSizeFinder()).startswith("MultiheadAttentionSizeFinder(")


def test_multihead_attention_size_finder_str() -> None:
    assert str(MultiheadAttentionSizeFinder()).startswith("MultiheadAttentionSizeFinder(")


def test_multihead_attention_size_finder_eq_true() -> None:
    assert MultiheadAttentionSizeFinder() == MultiheadAttentionSizeFinder()


def test_multihead_attention_size_finder_eq_false() -> None:
    assert MultiheadAttentionSizeFinder() != 42


@pytest.mark.parametrize("module", MULTIHEAD_ATTENTION_MODULES)
def test_multihead_attention_size_finder_find_in_features(module: ModuleSizes) -> None:
    assert MultiheadAttentionSizeFinder().find_in_features(module.module) == module.in_features


def test_multihead_attention_size_finder_find_in_features_incorrect() -> None:
    size_finder = MultiheadAttentionSizeFinder()
    module = nn.Identity()
    with pytest.raises(SizeNotFoundError, match=r"module .* does not have attribute embed_dim"):
        size_finder.find_in_features(module)


@pytest.mark.parametrize("module", MULTIHEAD_ATTENTION_MODULES)
def test_multihead_attention_size_finder_find_out_features(module: ModuleSizes) -> None:
    assert MultiheadAttentionSizeFinder().find_out_features(module.module) == module.out_features


def test_multihead_attention_size_finder_find_out_features_incorrect() -> None:
    size_finder = MultiheadAttentionSizeFinder()
    module = nn.Identity()
    with pytest.raises(SizeNotFoundError, match=r"module .* does not have attribute embed_dim"):
        size_finder.find_out_features(module)
