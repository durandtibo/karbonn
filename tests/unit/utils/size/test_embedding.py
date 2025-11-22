from __future__ import annotations

import pytest
from torch import nn

from karbonn.utils.size import EmbeddingSizeFinder
from karbonn.utils.size.base import SizeNotFoundError
from tests.unit.utils.size.utils import ModuleSizes

EMBEDDING_MODULES = [
    ModuleSizes(
        module=nn.Embedding(num_embeddings=5, embedding_dim=6), in_features=[1], out_features=[6]
    ),
    ModuleSizes(
        module=nn.Embedding(num_embeddings=5, embedding_dim=4), in_features=[1], out_features=[4]
    ),
    ModuleSizes(
        module=nn.EmbeddingBag(num_embeddings=5, embedding_dim=6), in_features=[1], out_features=[6]
    ),
    ModuleSizes(
        module=nn.EmbeddingBag(num_embeddings=5, embedding_dim=4), in_features=[1], out_features=[4]
    ),
]


#########################################
#     Tests for EmbeddingSizeFinder     #
#########################################


def test_embedding_size_finder_repr() -> None:
    assert repr(EmbeddingSizeFinder()).startswith("EmbeddingSizeFinder(")


def test_embedding_size_finder_str() -> None:
    assert str(EmbeddingSizeFinder()).startswith("EmbeddingSizeFinder(")


def test_embedding_size_finder_eq_true() -> None:
    assert EmbeddingSizeFinder() == EmbeddingSizeFinder()


def test_embedding_size_finder_eq_false() -> None:
    assert EmbeddingSizeFinder() != 42


@pytest.mark.parametrize("module", EMBEDDING_MODULES)
def test_embedding_size_finder_find_in_features(module: ModuleSizes) -> None:
    assert EmbeddingSizeFinder().find_in_features(module.module) == module.in_features


@pytest.mark.parametrize("module", EMBEDDING_MODULES)
def test_embedding_size_finder_find_out_features(module: ModuleSizes) -> None:
    assert EmbeddingSizeFinder().find_out_features(module.module) == module.out_features


def test_embedding_size_finder_find_out_features_incorrect() -> None:
    size_finder = EmbeddingSizeFinder()
    module = nn.Identity()
    with pytest.raises(SizeNotFoundError, match=r"module .* does not have attribute embedding_dim"):
        size_finder.find_out_features(module)
