from __future__ import annotations

from coola import objects_are_equal
from torch import nn

from karbonn.modules import ExU
from karbonn.utils.size import (
    BatchNormSizeFinder,
    BilinearSizeFinder,
    ConvolutionSizeFinder,
    EmbeddingSizeFinder,
    GroupNormSizeFinder,
    LinearSizeFinder,
    ModuleListSizeFinder,
    MultiheadAttentionSizeFinder,
    RecurrentSizeFinder,
    SequentialSizeFinder,
    TransformerLayerSizeFinder,
    TransformerSizeFinder,
    UnknownSizeFinder,
    get_karbonn_size_finders,
    get_size_finders,
    get_torch_size_finders,
)

######################################
#     Tests for get_size_finders     #
######################################


def test_get_size_finders() -> None:
    assert objects_are_equal(
        get_size_finders(),
        {
            nn.Module: UnknownSizeFinder(),
            nn.BatchNorm1d: BatchNormSizeFinder(),
            nn.BatchNorm2d: BatchNormSizeFinder(),
            nn.BatchNorm3d: BatchNormSizeFinder(),
            nn.GroupNorm: GroupNormSizeFinder(),
            nn.Bilinear: BilinearSizeFinder(),
            nn.Conv1d: ConvolutionSizeFinder(),
            nn.Conv2d: ConvolutionSizeFinder(),
            nn.Conv3d: ConvolutionSizeFinder(),
            nn.ConvTranspose1d: ConvolutionSizeFinder(),
            nn.ConvTranspose2d: ConvolutionSizeFinder(),
            nn.ConvTranspose3d: ConvolutionSizeFinder(),
            nn.Embedding: EmbeddingSizeFinder(),
            nn.EmbeddingBag: EmbeddingSizeFinder(),
            nn.GRU: RecurrentSizeFinder(),
            nn.LSTM: RecurrentSizeFinder(),
            nn.Linear: LinearSizeFinder(),
            nn.ModuleList: ModuleListSizeFinder(),
            nn.MultiheadAttention: MultiheadAttentionSizeFinder(),
            nn.RNN: RecurrentSizeFinder(),
            nn.Sequential: SequentialSizeFinder(),
            nn.SyncBatchNorm: BatchNormSizeFinder(),
            nn.TransformerDecoder: TransformerSizeFinder(),
            nn.TransformerDecoderLayer: TransformerLayerSizeFinder(),
            nn.TransformerEncoder: TransformerSizeFinder(),
            nn.TransformerEncoderLayer: TransformerLayerSizeFinder(),
            ExU: LinearSizeFinder(),
        },
    )


##############################################
#     Tests for get_karbonn_size_finders     #
##############################################


def test_get_karbonn_size_finders() -> None:
    assert objects_are_equal(get_karbonn_size_finders(), {ExU: LinearSizeFinder()})


############################################
#     Tests for get_torch_size_finders     #
############################################


def test_get_torch_size_finders() -> None:
    assert objects_are_equal(
        get_torch_size_finders(),
        {
            nn.Module: UnknownSizeFinder(),
            nn.BatchNorm1d: BatchNormSizeFinder(),
            nn.BatchNorm2d: BatchNormSizeFinder(),
            nn.BatchNorm3d: BatchNormSizeFinder(),
            nn.GroupNorm: GroupNormSizeFinder(),
            nn.Bilinear: BilinearSizeFinder(),
            nn.Conv1d: ConvolutionSizeFinder(),
            nn.Conv2d: ConvolutionSizeFinder(),
            nn.Conv3d: ConvolutionSizeFinder(),
            nn.ConvTranspose1d: ConvolutionSizeFinder(),
            nn.ConvTranspose2d: ConvolutionSizeFinder(),
            nn.ConvTranspose3d: ConvolutionSizeFinder(),
            nn.Embedding: EmbeddingSizeFinder(),
            nn.EmbeddingBag: EmbeddingSizeFinder(),
            nn.GRU: RecurrentSizeFinder(),
            nn.LSTM: RecurrentSizeFinder(),
            nn.Linear: LinearSizeFinder(),
            nn.ModuleList: ModuleListSizeFinder(),
            nn.MultiheadAttention: MultiheadAttentionSizeFinder(),
            nn.RNN: RecurrentSizeFinder(),
            nn.Sequential: SequentialSizeFinder(),
            nn.SyncBatchNorm: BatchNormSizeFinder(),
            nn.TransformerDecoder: TransformerSizeFinder(),
            nn.TransformerDecoderLayer: TransformerLayerSizeFinder(),
            nn.TransformerEncoder: TransformerSizeFinder(),
            nn.TransformerEncoderLayer: TransformerLayerSizeFinder(),
        },
    )
