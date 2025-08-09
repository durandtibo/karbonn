from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import pytest
import torch
from coola.utils.tensor import get_available_devices
from torch import nn

from karbonn.testing import tabulate_available
from karbonn.utils import freeze_module
from karbonn.utils.summary.module import (
    ModuleSummary,
    get_in_dtype,
    get_in_size,
    get_layer_names,
    get_layer_types,
    get_num_learnable_parameters,
    get_num_parameters,
    get_out_dtype,
    get_out_size,
    merge_size_dtype,
    module_summary,
    multiline_format,
    parse_batch_dtype,
    parse_batch_shape,
    str_module_summary,
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


@pytest.fixture(scope="module")
def nested_module() -> nn.Module:
    return nn.Sequential(
        nn.Linear(4, 6),
        nn.ReLU(),
        nn.Sequential(
            nn.Linear(6, 6),
            nn.Dropout(0.1),
            nn.Sequential(nn.Linear(6, 6), nn.PReLU()),
            nn.Linear(6, 3),
        ),
    )


@pytest.fixture(scope="module")
def summary(nested_module: nn.Module) -> dict[str, ModuleSummary]:
    return module_summary(nested_module, depth=2, input_args=[torch.randn(2, 4)])


###################################
#     Tests for ModuleSummary     #
###################################


def test_module_summary_class_repr() -> None:
    summary = ModuleSummary(nn.Linear(4, 6))
    assert repr(summary).startswith("ModuleSummary(")


def test_module_summary_class_repr_forward() -> None:
    module = nn.Linear(4, 6)
    module(torch.rand(2, 4))
    summary = ModuleSummary(module)
    assert repr(summary).startswith("ModuleSummary(")


def test_module_summary_class_str() -> None:
    summary = ModuleSummary(nn.Linear(4, 6))
    assert str(summary).startswith("ModuleSummary(")


def test_module_summary_class_str_forward() -> None:
    module = nn.Linear(4, 6)
    module(torch.rand(2, 4))
    summary = ModuleSummary(module)
    assert str(summary).startswith("ModuleSummary(")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("input_size", SIZES)
@pytest.mark.parametrize("output_size", SIZES)
def test_module_summary_class_linear(
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
def test_module_summary_class_bilinear(device: str) -> None:
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
def test_module_summary_class_linear_no_forward(
    device: str, input_size: int, output_size: int
) -> None:
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
def test_module_summary_class_gru(
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
def test_module_summary_class_module_dict(
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
def test_module_summary_class_custom_module(device: str, batch_size: int) -> None:
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


def test_module_summary_class_detach() -> None:
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


####################################
#     Tests for module_summary     #
####################################


@pytest.mark.parametrize("depth", [-1, 0, 1, 2, 3])
def test_module_summary_linear(depth: int) -> None:
    linear = nn.Linear(4, 6)
    summary = module_summary(linear, depth=depth)
    assert len(summary) == 1
    assert isinstance(summary["[root]"], ModuleSummary)
    assert summary["[root]"].module == linear


def test_module_summary_linear_input_args() -> None:
    linear = nn.Linear(4, 6)
    summary = module_summary(linear, input_args=[torch.randn(2, 4)])
    assert len(summary) == 1
    assert isinstance(summary["[root]"], ModuleSummary)
    assert summary["[root]"].module == linear


def test_module_summary_linear_input_kwargs() -> None:
    linear = nn.Linear(4, 6)
    summary = module_summary(linear, input_kwargs={"input": torch.randn(2, 4)})
    assert len(summary) == 1
    assert isinstance(summary["[root]"], ModuleSummary)
    assert summary["[root]"].module == linear


def test_module_summary_depth_0_sequential(nested_module: nn.Module) -> None:
    summary = module_summary(nested_module)
    assert len(summary) == 1
    assert isinstance(summary["[root]"], ModuleSummary)
    assert summary["[root]"].module == nested_module


def test_module_summary_depth_1_sequential(nested_module: nn.Module) -> None:
    summary = module_summary(nested_module, depth=1)
    assert len(summary) == 4
    assert isinstance(summary["[root]"], ModuleSummary)
    assert summary["[root]"].module == nested_module
    assert isinstance(summary["0"], ModuleSummary)
    assert summary["0"].module == nested_module[0]
    assert isinstance(summary["1"], ModuleSummary)
    assert summary["1"].module == nested_module[1]
    assert isinstance(summary["2"], ModuleSummary)
    assert summary["2"].module == nested_module[2]


def test_module_summary_depth_2_sequential(nested_module: nn.Module) -> None:
    summary = module_summary(nested_module, depth=2)
    assert len(summary) == 8
    assert isinstance(summary["[root]"], ModuleSummary)
    assert summary["[root]"].module == nested_module
    assert isinstance(summary["0"], ModuleSummary)
    assert summary["0"].module == nested_module[0]
    assert isinstance(summary["1"], ModuleSummary)
    assert summary["1"].module == nested_module[1]
    assert isinstance(summary["2"], ModuleSummary)
    assert summary["2"].module == nested_module[2]
    assert isinstance(summary["2.0"], ModuleSummary)
    assert summary["2.0"].module == nested_module[2][0]
    assert isinstance(summary["2.1"], ModuleSummary)
    assert summary["2.1"].module == nested_module[2][1]
    assert isinstance(summary["2.2"], ModuleSummary)
    assert summary["2.2"].module == nested_module[2][2]
    assert isinstance(summary["2.3"], ModuleSummary)
    assert summary["2.3"].module == nested_module[2][3]


########################################
#     Tests for str_module_summary     #
########################################


@tabulate_available
@pytest.mark.parametrize("depth", [-1, 0, 1, 2, 3])
def test_str_module_summary_linear(depth: int) -> None:
    out = str_module_summary(module_summary(nn.Linear(4, 6), depth=depth))
    assert out == (
        "╒════╤════════╤════════╤══════════════════╤═══════════════════╤════════════════════╕\n"
        "│    │ name   │ type   │ params (learn)   │ in size (dtype)   │ out size (dtype)   │\n"
        "╞════╪════════╪════════╪══════════════════╪═══════════════════╪════════════════════╡\n"
        "│  0 │ [root] │ Linear │ 30 (30)          │ ? (?)             │ ? (?)              │\n"
        "╘════╧════════╧════════╧══════════════════╧═══════════════════╧════════════════════╛"
    )


@tabulate_available
@pytest.mark.parametrize("depth", [-1, 0, 1, 2, 3])
def test_str_module_summary_linear_forward(depth: int) -> None:
    out = str_module_summary(
        module_summary(nn.Linear(4, 6), depth=depth, input_args=[torch.randn(2, 4)])
    )
    assert out == (
        "╒════╤════════╤════════╤══════════════════╤════════════════════════╤════════════════════════╕\n"
        "│    │ name   │ type   │ params (learn)   │ in size (dtype)        │ out size (dtype)       │\n"
        "╞════╪════════╪════════╪══════════════════╪════════════════════════╪════════════════════════╡\n"
        "│  0 │ [root] │ Linear │ 30 (30)          │ [2, 4] (torch.float32) │ [2, 6] (torch.float32) │\n"
        "╘════╧════════╧════════╧══════════════════╧════════════════════════╧════════════════════════╛"
    )


@tabulate_available
def test_str_module_summary_nested_depth_0(nested_module: nn.Module) -> None:
    out = str_module_summary(module_summary(nested_module, input_args=[torch.randn(2, 4)]))
    assert out == (
        "╒════╤════════╤════════════╤══════════════════╤════════════════════════╤════════════════════════╕\n"
        "│    │ name   │ type       │ params (learn)   │ in size (dtype)        │ out size (dtype)       │\n"
        "╞════╪════════╪════════════╪══════════════════╪════════════════════════╪════════════════════════╡\n"
        "│  0 │ [root] │ Sequential │ 136 (136)        │ [2, 4] (torch.float32) │ [2, 3] (torch.float32) │\n"
        "╘════╧════════╧════════════╧══════════════════╧════════════════════════╧════════════════════════╛"
    )


@tabulate_available
def test_str_module_summary_nested_depth_1(nested_module: nn.Module) -> None:
    out = str_module_summary(module_summary(nested_module, depth=1, input_args=[torch.randn(2, 4)]))
    assert out == (
        "╒════╤════════╤════════════╤══════════════════╤════════════════════════╤════════════════════════╕\n"
        "│    │ name   │ type       │ params (learn)   │ in size (dtype)        │ out size (dtype)       │\n"
        "╞════╪════════╪════════════╪══════════════════╪════════════════════════╪════════════════════════╡\n"
        "│  0 │ [root] │ Sequential │ 136 (136)        │ [2, 4] (torch.float32) │ [2, 3] (torch.float32) │\n"
        "├────┼────────┼────────────┼──────────────────┼────────────────────────┼────────────────────────┤\n"
        "│  1 │ 0      │ Linear     │ 30 (30)          │ [2, 4] (torch.float32) │ [2, 6] (torch.float32) │\n"
        "├────┼────────┼────────────┼──────────────────┼────────────────────────┼────────────────────────┤\n"
        "│  2 │ 1      │ ReLU       │ 0 (0)            │ [2, 6] (torch.float32) │ [2, 6] (torch.float32) │\n"
        "├────┼────────┼────────────┼──────────────────┼────────────────────────┼────────────────────────┤\n"
        "│  3 │ 2      │ Sequential │ 106 (106)        │ [2, 6] (torch.float32) │ [2, 3] (torch.float32) │\n"
        "╘════╧════════╧════════════╧══════════════════╧════════════════════════╧════════════════════════╛"
    )


@tabulate_available
def test_str_module_summary_nested_depth_2(nested_module: nn.Module) -> None:
    out = str_module_summary(module_summary(nested_module, depth=2, input_args=[torch.randn(2, 4)]))
    assert out == (
        "╒════╤════════╤════════════╤══════════════════╤════════════════════════╤════════════════════════╕\n"
        "│    │ name   │ type       │ params (learn)   │ in size (dtype)        │ out size (dtype)       │\n"
        "╞════╪════════╪════════════╪══════════════════╪════════════════════════╪════════════════════════╡\n"
        "│  0 │ [root] │ Sequential │ 136 (136)        │ [2, 4] (torch.float32) │ [2, 3] (torch.float32) │\n"
        "├────┼────────┼────────────┼──────────────────┼────────────────────────┼────────────────────────┤\n"
        "│  1 │ 0      │ Linear     │ 30 (30)          │ [2, 4] (torch.float32) │ [2, 6] (torch.float32) │\n"
        "├────┼────────┼────────────┼──────────────────┼────────────────────────┼────────────────────────┤\n"
        "│  2 │ 1      │ ReLU       │ 0 (0)            │ [2, 6] (torch.float32) │ [2, 6] (torch.float32) │\n"
        "├────┼────────┼────────────┼──────────────────┼────────────────────────┼────────────────────────┤\n"
        "│  3 │ 2      │ Sequential │ 106 (106)        │ [2, 6] (torch.float32) │ [2, 3] (torch.float32) │\n"
        "├────┼────────┼────────────┼──────────────────┼────────────────────────┼────────────────────────┤\n"
        "│  4 │ 2.0    │ Linear     │ 42 (42)          │ [2, 6] (torch.float32) │ [2, 6] (torch.float32) │\n"
        "├────┼────────┼────────────┼──────────────────┼────────────────────────┼────────────────────────┤\n"
        "│  5 │ 2.1    │ Dropout    │ 0 (0)            │ [2, 6] (torch.float32) │ [2, 6] (torch.float32) │\n"
        "├────┼────────┼────────────┼──────────────────┼────────────────────────┼────────────────────────┤\n"
        "│  6 │ 2.2    │ Sequential │ 43 (43)          │ [2, 6] (torch.float32) │ [2, 6] (torch.float32) │\n"
        "├────┼────────┼────────────┼──────────────────┼────────────────────────┼────────────────────────┤\n"
        "│  7 │ 2.3    │ Linear     │ 21 (21)          │ [2, 6] (torch.float32) │ [2, 3] (torch.float32) │\n"
        "╘════╧════════╧════════════╧══════════════════╧════════════════════════╧════════════════════════╛"
    )


######################################
#     Tests for multiline_format     #
######################################


def test_multiline_format_empty() -> None:
    assert multiline_format([]) == []


@pytest.mark.parametrize(
    "data",
    [
        [
            "torch.float32",
            ["torch.float32", "torch.int32"],
            {"input1": "torch.float32", "input2": "torch.int64"},
        ],
        (
            "torch.float32",
            ["torch.float32", "torch.int32"],
            {"input1": "torch.float32", "input2": "torch.int64"},
        ),
        (
            None,
            "torch.float32",
            None,
            ["torch.float32", "torch.int32"],
            {"input1": "torch.float32", "input2": "torch.int64"},
            None,
        ),
    ],
)
def test_multiline_format_list(data: Sequence) -> None:
    assert multiline_format(data) == [
        "torch.float32",
        "(0): torch.float32\n(1): torch.int32",
        "(input1): torch.float32\n(input2): torch.int64",
    ]


#####################################
#     Tests for get_layer_names     #
#####################################


def test_get_layer_names(summary: dict[str, ModuleSummary]) -> None:
    assert get_layer_names(summary) == ["[root]", "0", "1", "2", "2.0", "2.1", "2.2", "2.3"]


def test_get_layer_names_empty() -> None:
    assert get_layer_names({}) == []


#####################################
#     Tests for get_layer_types     #
#####################################


def test_get_layer_types(summary: dict[str, ModuleSummary]) -> None:
    assert get_layer_types(summary) == [
        "Sequential",
        "Linear",
        "ReLU",
        "Sequential",
        "Linear",
        "Dropout",
        "Sequential",
        "Linear",
    ]


def test_get_layer_types_empty() -> None:
    assert get_layer_types({}) == []


########################################
#     Tests for get_num_parameters     #
########################################


def test_get_num_parameters(summary: dict[str, ModuleSummary]) -> None:
    assert get_num_parameters(summary) == [136, 30, 0, 106, 42, 0, 43, 21]


def test_get_num_parameters_empty() -> None:
    assert get_num_parameters({}) == []


##################################################
#     Tests for get_num_learnable_parameters     #
##################################################


def test_get_num_learnable_parameters(summary: dict[str, ModuleSummary]) -> None:
    assert get_num_learnable_parameters(summary) == [136, 30, 0, 106, 42, 0, 43, 21]


def test_get_num_learnable_parameters_empty() -> None:
    assert get_num_learnable_parameters({}) == []


##################################
#     Tests for get_in_dtype     #
##################################


def test_get_in_dtype(summary: dict[str, ModuleSummary]) -> None:
    assert get_in_dtype(summary) == [
        torch.float,
        torch.float,
        torch.float,
        torch.float,
        torch.float,
        torch.float,
        torch.float,
        torch.float,
    ]


def test_get_in_dtype_empty() -> None:
    assert get_in_dtype({}) == []


###################################
#     Tests for get_out_dtype     #
###################################


def test_get_out_dtype(summary: dict[str, ModuleSummary]) -> None:
    assert get_out_dtype(summary) == [
        torch.float,
        torch.float,
        torch.float,
        torch.float,
        torch.float,
        torch.float,
        torch.float,
        torch.float,
    ]


def test_get_out_dtype_empty() -> None:
    assert get_out_dtype({}) == []


#################################
#     Tests for get_in_size     #
#################################


def test_get_in_size(summary: dict[str, ModuleSummary]) -> None:
    assert get_in_size(summary) == [
        torch.Size([2, 4]),
        torch.Size([2, 4]),
        torch.Size([2, 6]),
        torch.Size([2, 6]),
        torch.Size([2, 6]),
        torch.Size([2, 6]),
        torch.Size([2, 6]),
        torch.Size([2, 6]),
    ]


def test_get_in_size_empty() -> None:
    assert get_in_size({}) == []


##################################
#     Tests for get_out_size     #
##################################


def test_get_out_size(summary: dict[str, ModuleSummary]) -> None:
    assert get_out_size(summary) == [
        torch.Size([2, 3]),
        torch.Size([2, 6]),
        torch.Size([2, 6]),
        torch.Size([2, 3]),
        torch.Size([2, 6]),
        torch.Size([2, 6]),
        torch.Size([2, 6]),
        torch.Size([2, 3]),
    ]


def test_get_out_size_empty() -> None:
    assert get_out_size({}) == []


######################################
#     Tests for merge_size_dtype     #
######################################


def test_merge_size_dtype() -> None:
    output = merge_size_dtype(
        sizes=[
            torch.Size([2, 3]),
            [torch.Size([2, 4]), torch.Size([2, 5]), torch.Size([2, 6, 3])],
            {"input1": torch.Size([2, 4]), "input2": torch.Size([2, 3, 4])},
            None,
        ],
        dtypes=[
            torch.float32,
            [torch.float32, torch.long, torch.float],
            {"input1": torch.long, "input2": torch.float32},
            None,
        ],
    )
    assert output == [
        "[2, 3] (torch.float32)",
        "(0): [2, 4] (torch.float32)\n(1): [2, 5] (torch.int64)\n(2): [2, 6, 3] (torch.float32)",
        "(input1): [2, 4] (torch.int64)\n(input2): [2, 3, 4] (torch.float32)",
        "? (?)",
    ]
