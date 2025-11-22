from __future__ import annotations

import pytest
from torch import nn

from karbonn.utils.size import TransformerLayerSizeFinder, TransformerSizeFinder
from karbonn.utils.size.base import SizeNotFoundError
from tests.unit.utils.size.utils import ModuleSizes

TRANFORMER_LAYERS_MODULES = [
    ModuleSizes(
        module=nn.TransformerEncoderLayer(d_model=4, nhead=1), in_features=[4], out_features=[4]
    ),
    ModuleSizes(
        module=nn.TransformerEncoderLayer(d_model=8, nhead=2), in_features=[8], out_features=[8]
    ),
    ModuleSizes(
        module=nn.TransformerDecoderLayer(d_model=4, nhead=1), in_features=[4], out_features=[4]
    ),
    ModuleSizes(
        module=nn.TransformerDecoderLayer(d_model=8, nhead=2), in_features=[8], out_features=[8]
    ),
]

TRANFORMER_MODULES = [
    ModuleSizes(
        module=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=4, nhead=1), num_layers=1),
        in_features=[4],
        out_features=[4],
    ),
    ModuleSizes(
        module=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=8, nhead=2), num_layers=1),
        in_features=[8],
        out_features=[8],
    ),
    ModuleSizes(
        module=nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=4, nhead=1), num_layers=1),
        in_features=[4],
        out_features=[4],
    ),
    ModuleSizes(
        module=nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=8, nhead=2), num_layers=1),
        in_features=[8],
        out_features=[8],
    ),
]


################################################
#     Tests for TransformerLayerSizeFinder     #
################################################


def test_transformer_layer_size_finder_repr() -> None:
    assert repr(TransformerLayerSizeFinder()).startswith("TransformerLayerSizeFinder(")


def test_transformer_layer_size_finder_str() -> None:
    assert str(TransformerLayerSizeFinder()).startswith("TransformerLayerSizeFinder(")


def test_transformer_layer_size_finder_eq_true() -> None:
    assert TransformerLayerSizeFinder() == TransformerLayerSizeFinder()


def test_transformer_layer_size_finder_eq_false() -> None:
    assert TransformerLayerSizeFinder() != 42


@pytest.mark.parametrize("module", TRANFORMER_LAYERS_MODULES)
def test_transformer_layer_size_finder_find_in_features(module: ModuleSizes) -> None:
    assert TransformerLayerSizeFinder().find_in_features(module.module) == module.in_features


def test_transformer_layer_size_finder_find_in_features_incorrect() -> None:
    size_finder = TransformerLayerSizeFinder()
    module = nn.Identity()
    with pytest.raises(SizeNotFoundError, match=r"module .* does not have attribute self_attn"):
        size_finder.find_in_features(module)


@pytest.mark.parametrize("module", TRANFORMER_LAYERS_MODULES)
def test_transformer_layer_size_finder_find_out_features(module: ModuleSizes) -> None:
    assert TransformerLayerSizeFinder().find_out_features(module.module) == module.out_features


def test_transformer_layer_size_finder_find_out_features_incorrect() -> None:
    size_finder = TransformerLayerSizeFinder()
    module = nn.Identity()
    with pytest.raises(SizeNotFoundError, match=r"module .* does not have attribute self_attn"):
        size_finder.find_out_features(module)


###########################################
#     Tests for TransformerSizeFinder     #
###########################################


def test_transformer_size_finder_repr() -> None:
    assert repr(TransformerSizeFinder()).startswith("TransformerSizeFinder(")


def test_transformer_size_finder_str() -> None:
    assert str(TransformerSizeFinder()).startswith("TransformerSizeFinder(")


def test_transformer_size_finder_eq_true() -> None:
    assert TransformerSizeFinder() == TransformerSizeFinder()


def test_transformer_size_finder_eq_false() -> None:
    assert TransformerSizeFinder() != 42


@pytest.mark.parametrize("module", TRANFORMER_MODULES)
def test_transformer_size_finder_find_in_features(module: ModuleSizes) -> None:
    assert TransformerSizeFinder().find_in_features(module.module) == module.in_features


def test_transformer_size_finder_find_in_features_incorrect() -> None:
    size_finder = TransformerSizeFinder()
    module = nn.Identity()
    with pytest.raises(SizeNotFoundError, match=r"module .* does not have attribute layers"):
        size_finder.find_in_features(module)


@pytest.mark.parametrize("module", TRANFORMER_MODULES)
def test_transformer_size_finder_find_out_features(module: ModuleSizes) -> None:
    assert TransformerSizeFinder().find_out_features(module.module) == module.out_features


def test_transformer_size_finder_find_out_features_incorrect() -> None:
    size_finder = TransformerSizeFinder()
    module = nn.Identity()
    with pytest.raises(SizeNotFoundError, match=r"module .* does not have attribute layers"):
        size_finder.find_out_features(module)
