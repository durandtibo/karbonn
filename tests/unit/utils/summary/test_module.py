from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import pytest
import torch
from coola.utils.tensor import get_available_devices
from torch import nn

from karbonn.utils import freeze_module
from karbonn.utils.summary.module import (
    ModuleSummary,
    parse_batch_dtype,
    parse_batch_shape,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SIZES = [1, 2, 3]


class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=8)
        # The parameters of the embedding layer should not appear in the learnable parameters.
        freeze_module(self.embedding)
        self.fc = nn.Linear(8, 4)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embedding(tensor))


class MyModuleDict(nn.Module):

    def __init__(self, in_features: int = 4, out_features: int = 6) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.criterion = nn.MSELoss()

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        prediction = self.linear(batch["input"])
        return {"loss": self.criterion(prediction, batch["target"]), "prediction": prediction}


###################################
#     Tests for ModuleSummary     #
###################################


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("input_size", SIZES)
@pytest.mark.parametrize("output_size", SIZES)
def test_module_summary_linear(
    device: str, batch_size: int, input_size: int, output_size: int
) -> None:
    device = torch.device(device)
    module = nn.Linear(input_size, output_size).to(device=device)
    summary = ModuleSummary(module)
    assert summary.get_num_parameters() == input_size * output_size + output_size
    assert summary.get_num_learnable_parameters() == input_size * output_size + output_size
    assert summary.get_layer_type() == "Linear"

    # Run the forward to capture the input and output sizes.
    module(torch.rand(batch_size, input_size, device=device))
    assert summary.get_in_size() == torch.Size([batch_size, input_size])
    assert summary.get_out_size() == torch.Size([batch_size, output_size])
    assert summary.get_in_dtype() == torch.float32
    assert summary.get_out_dtype() == torch.float32


@pytest.mark.parametrize("device", get_available_devices())
def test_module_summary_bilinear(device: str) -> None:
    device = torch.device(device)
    module = nn.Bilinear(in1_features=3, in2_features=4, out_features=7).to(device=device)
    summary = ModuleSummary(module)
    assert summary.get_num_parameters() == 91
    assert summary.get_num_learnable_parameters() == 91
    assert summary.get_layer_type() == "Bilinear"

    # Run the forward to capture the input and output sizes.
    module(torch.rand(2, 3, device=device), torch.rand(2, 4, device=device))
    assert summary.get_in_size() == ((2, 3), (2, 4))
    assert summary.get_out_size() == (2, 7)
    assert summary.get_in_dtype() == (torch.float32, torch.float32)
    assert summary.get_out_dtype() == torch.float32


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("input_size", SIZES)
@pytest.mark.parametrize("output_size", SIZES)
def test_module_summary_linear_no_forward(device: str, input_size: int, output_size: int) -> None:
    device = torch.device(device)
    module = nn.Linear(input_size, output_size).to(device=device)
    summary = ModuleSummary(module)
    assert summary.get_num_parameters() == input_size * output_size + output_size
    assert summary.get_num_learnable_parameters() == input_size * output_size + output_size
    assert summary.get_layer_type() == "Linear"
    assert summary.get_in_size() is None
    assert summary.get_out_size() is None
    assert summary.get_in_dtype() is None
    assert summary.get_out_dtype() is None


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("input_size", SIZES)
@pytest.mark.parametrize("hidden_size", SIZES)
def test_module_summary_gru(
    device: str, batch_size: int, seq_len: int, input_size: int, hidden_size: int
) -> None:
    device = torch.device(device)
    module = nn.GRU(input_size, hidden_size).to(device=device)
    summary = ModuleSummary(module)
    num_parameters = 3 * ((input_size + 1) * hidden_size + (hidden_size + 1) * hidden_size)
    assert summary.get_num_parameters() == num_parameters
    assert summary.get_num_learnable_parameters() == num_parameters
    assert summary.get_layer_type() == "GRU"

    # Run the forward to capture the input and output sizes.
    module(torch.rand(seq_len, batch_size, input_size, device=device))
    assert summary.get_in_size() == (seq_len, batch_size, input_size)
    assert summary.get_out_size() == (
        (seq_len, batch_size, hidden_size),
        (1, batch_size, hidden_size),
    )
    assert summary.get_in_dtype() == torch.float32
    assert summary.get_out_dtype() == (torch.float32, torch.float32)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("input_size", SIZES)
@pytest.mark.parametrize("output_size", SIZES)
def test_module_summary_module_dict(
    device: str, batch_size: int, input_size: int, output_size: int
) -> None:
    device = torch.device(device)
    module = MyModuleDict(in_features=input_size, out_features=output_size).to(device=device)
    summary = ModuleSummary(module)
    assert summary.get_num_parameters() == input_size * output_size + output_size
    assert summary.get_num_learnable_parameters() == input_size * output_size + output_size
    assert summary.get_layer_type() == "MyModuleDict"

    # Run the forward to capture the input and output sizes.
    module(
        {
            "input": torch.rand(batch_size, input_size, device=device),
            "target": torch.rand(batch_size, output_size, device=device),
        }
    )
    assert summary.get_in_size() == {
        "input": torch.Size([batch_size, input_size]),
        "target": torch.Size([batch_size, output_size]),
    }
    assert summary.get_out_size() == {
        "loss": torch.Size([]),
        "prediction": torch.Size([batch_size, output_size]),
    }
    assert summary.get_in_dtype() == {"input": torch.float32, "target": torch.float32}
    assert summary.get_out_dtype() == {"loss": torch.float32, "prediction": torch.float32}


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
def test_module_summary_custom_module(device: str, batch_size: int) -> None:
    device = torch.device(device)
    module = MyModule().to(device=device)
    summary = ModuleSummary(module)
    assert summary.get_num_parameters() == 116
    assert summary.get_num_learnable_parameters() == 36
    assert summary.get_layer_type() == "MyModule"

    # Run the forward to capture the input and output sizes.
    module(torch.ones(batch_size, dtype=torch.long, device=device))
    assert summary.get_in_size() == (batch_size,)
    assert summary.get_out_size() == (batch_size, 4)
    assert summary.get_in_dtype() == torch.int64
    assert summary.get_out_dtype() == torch.float32


def test_module_summary_detach() -> None:
    module = nn.Linear(4, 6)
    summary = ModuleSummary(module)
    assert summary._hook_handle.id in module._forward_hooks
    summary.detach_hook()
    assert summary._hook_handle.id not in module._forward_hooks


#######################################
#     Tests for parse_batch_dtype     #
#######################################


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.long,
        torch.float,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.cdouble,
        torch.complex64,
    ],
)
def test_parse_batch_dtype_tensor(dtype: torch.dtype) -> None:
    assert parse_batch_dtype(torch.ones(2, 3, dtype=dtype)) == dtype


@pytest.mark.parametrize(
    "batch",
    [
        (torch.ones(2, 3), torch.ones(2, dtype=torch.long)),
        [torch.ones(2, 3), torch.ones(2, dtype=torch.long)],
    ],
)
def test_parse_batch_dtype_sequence(batch: Sequence) -> None:
    assert parse_batch_dtype(batch) == (torch.float32, torch.int64)


def test_parse_batch_dtype_sequence_empty() -> None:
    assert parse_batch_dtype([]) == ()


@pytest.mark.parametrize(
    "batch",
    [
        {"feature1": torch.ones(2, 3), "feature2": torch.ones(2, dtype=torch.long)},
        OrderedDict({"feature1": torch.ones(2, 3), "feature2": torch.ones(2, dtype=torch.long)}),
    ],
)
def test_parse_batch_dtype_dict(batch: Mapping[str, torch.Tensor]) -> None:
    assert parse_batch_dtype(batch) == {"feature1": torch.float32, "feature2": torch.long}


def test_parse_batch_dtype_dict_empty() -> None:
    assert parse_batch_dtype({}) == {}


def test_parse_batch_dtype_invalid() -> None:
    assert parse_batch_dtype(set()) is None


def test_parse_batch_dtype_nested() -> None:
    assert parse_batch_dtype(
        (torch.ones(2, 3), (torch.ones(2, dtype=torch.long), torch.ones(2, 3)))
    ) == (
        torch.float32,
        (torch.int64, torch.float32),
    )


#######################################
#     Tests for parse_batch_shape     #
#######################################


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("input_size", SIZES)
def test_parse_batch_shape_tensor_2d(batch_size: int, input_size: int) -> None:
    assert parse_batch_shape(torch.ones(batch_size, input_size)) == torch.Size(
        [batch_size, input_size]
    )


@pytest.mark.parametrize(("batch_size", "seq_len", "input_size"), [(1, 1, 1), (2, 2, 2)])
def test_parse_batch_shape_tensor_3d(batch_size: int, seq_len: int, input_size: int) -> None:
    assert parse_batch_shape(torch.ones(batch_size, seq_len, input_size)) == torch.Size(
        [batch_size, seq_len, input_size]
    )


@pytest.mark.parametrize(
    "batch",
    [
        (torch.ones(2, 3), torch.ones(2, dtype=torch.long)),
        [torch.ones(2, 3), torch.ones(2, dtype=torch.long)],
    ],
)
def test_parse_batch_shape_sequence(batch: Sequence[torch.Tensor]) -> None:
    assert parse_batch_shape(batch) == (torch.Size([2, 3]), torch.Size([2]))


def test_parse_batch_shape_sequence_empty() -> None:
    assert parse_batch_shape([]) == ()


@pytest.mark.parametrize(
    "batch",
    [
        {"feature1": torch.ones(2, 3), "feature2": torch.ones(2, dtype=torch.long)},
        OrderedDict({"feature1": torch.ones(2, 3), "feature2": torch.ones(2, dtype=torch.long)}),
    ],
)
def test_parse_batch_shape_dict(batch: Mapping[str, torch.Tensor]) -> None:
    assert parse_batch_shape(batch) == {"feature1": torch.Size([2, 3]), "feature2": torch.Size([2])}


def test_parse_batch_shape_dict_empty() -> None:
    assert parse_batch_shape({}) == {}


def test_parse_batch_shape_invalid() -> None:
    assert parse_batch_shape(set()) is None


def test_parse_batch_shape_nested() -> None:
    assert parse_batch_shape(
        (torch.ones(2, 3), (torch.ones(2, dtype=torch.long), torch.ones(2, 3)))
    ) == (
        torch.Size([2, 3]),
        (torch.Size([2]), torch.Size([2, 3])),
    )
