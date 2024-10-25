from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices

from karbonn.modules import CosSinNumericalEncoder
from karbonn.modules.numerical.sine import prepare_tensor_param

SIZES = (1, 2, 3)


############################################
#     Tests for CosSinNumericalEncoder     #
############################################

# COSSIN_MODULE_CONSTRUCTORS: tuple[Callable, ...] = (
#     CosSinNumericalEncoder.create_rand_frequency,
#     CosSinNumericalEncoder.create_linspace_frequency,
#     CosSinNumericalEncoder.create_logspace_frequency,
# )


def test_cos_sin_numerical_encoder_str() -> None:
    assert str(
        CosSinNumericalEncoder(
            frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
            phase_shift=torch.zeros(2, 3),
        )
    ).startswith("CosSinNumericalEncoder(")


@pytest.mark.parametrize(
    "frequency",
    [torch.tensor([1.0, 2.0, 4.0]), torch.tensor([[1.0, 2.0, 4.0]])],
)
def test_cos_sin_numerical_encoder_frequency_1_feature(frequency: torch.Tensor) -> None:
    assert CosSinNumericalEncoder(
        frequency=frequency, phase_shift=torch.zeros(1, 3)
    ).frequency.equal(torch.tensor([[1.0, 2.0, 4.0, 1.0, 2.0, 4.0]]))


def test_cos_sin_numerical_encoder_frequency_2_features() -> None:
    assert CosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
        phase_shift=torch.zeros(2, 3),
    ).frequency.equal(
        torch.tensor([[1.0, 2.0, 4.0, 1.0, 2.0, 4.0], [2.0, 3.0, 4.0, 2.0, 3.0, 4.0]])
    )


@pytest.mark.parametrize(
    "phase_shift",
    [torch.tensor([2.0, 1.0, 0.0]), torch.tensor([[2.0, 1.0, 0.0]])],
)
def test_cos_sin_numerical_encoder_phase_shift_1_feature(phase_shift: torch.Tensor) -> None:
    assert CosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 4.0]]), phase_shift=phase_shift
    ).phase_shift.equal(torch.tensor([[2.0, 1.0, 0.0, 2.0, 1.0, 0.0]]))


def test_cos_sin_numerical_encoder_phase_shift_2_features() -> None:
    assert CosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
        phase_shift=torch.tensor([[2.0, 1.0, 0.0], [1.0, 2.0, 3.0]]),
    ).phase_shift.equal(
        torch.tensor([[2.0, 1.0, 0.0, 2.0, 1.0, 0.0], [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]])
    )


def test_cos_sin_numerical_encoder_different_shape() -> None:
    with pytest.raises(RuntimeError, match="'frequency' and 'phase_shift' shapes do not match:"):
        CosSinNumericalEncoder(
            frequency=torch.ones(2, 4),
            phase_shift=torch.zeros(2, 3),
        )


def test_cos_sin_numerical_encoder_input_size() -> None:
    assert (
        CosSinNumericalEncoder(
            frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
            phase_shift=torch.zeros(2, 3),
        ).input_size
        == 2
    )


def test_cos_sin_numerical_encoder_output_size() -> None:
    assert (
        CosSinNumericalEncoder(
            frequency=torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
            phase_shift=torch.zeros(2, 3),
        ).output_size
        == 6
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("n_features", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_cos_sin_numerical_encoder_forward_2d(
    device: str, batch_size: int, n_features: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = CosSinNumericalEncoder(
        frequency=torch.rand(n_features, feature_size),
        phase_shift=torch.rand(n_features, feature_size),
    ).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, n_features, device=device))
    assert out.shape == (batch_size, n_features, feature_size * 2)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("n_features", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_cos_sin_numerical_encoder_forward_3d_batch_first(
    device: str, batch_size: int, seq_len: int, n_features: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = CosSinNumericalEncoder(
        frequency=torch.rand(n_features, feature_size),
        phase_shift=torch.rand(n_features, feature_size),
    ).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, seq_len, n_features, device=device))
    assert out.shape == (batch_size, seq_len, n_features, feature_size * 2)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("n_features", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_cos_sin_numerical_encoder_forward_3d_seq_first(
    device: str, batch_size: int, seq_len: int, n_features: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = CosSinNumericalEncoder(
        frequency=torch.rand(n_features, feature_size),
        phase_shift=torch.rand(n_features, feature_size),
    ).to(device=device)
    module.train(mode)
    out = module(torch.rand(seq_len, batch_size, n_features, device=device))
    assert out.shape == (seq_len, batch_size, n_features, feature_size * 2)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("learnable", [True, False])
@pytest.mark.parametrize("mode", [True, False])
def test_cos_sin_numerical_encoder_backward(
    device: str, batch_size: int, learnable: bool, mode: bool
) -> None:
    device = torch.device(device)
    module = CosSinNumericalEncoder(
        frequency=torch.rand(3, 6), phase_shift=torch.rand(3, 6), learnable=learnable
    ).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, 3, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, 3, 12)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("learnable", [True, False])
def test_cos_sin_numerical_encoder_learnable(learnable: bool) -> None:
    module = CosSinNumericalEncoder(
        frequency=torch.rand(3, 6), phase_shift=torch.rand(3, 6), learnable=learnable
    )
    assert module.frequency.requires_grad == learnable
    assert module.phase_shift.requires_grad == learnable


def test_cos_sin_numerical_encoder_forward_1_feature() -> None:
    module = CosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 3.0]]),
        phase_shift=torch.zeros(1, 3),
    )
    assert objects_are_allclose(
        module(torch.tensor([[-1], [0], [1]])),
        torch.tensor(
            [
                [
                    [
                        -0.8414709848078965,
                        -0.9092974268256817,
                        -0.1411200080598672,
                        0.5403023058681398,
                        -0.4161468365471424,
                        -0.9899924966004454,
                    ]
                ],
                [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
                [
                    [
                        0.8414709848078965,
                        0.9092974268256817,
                        0.1411200080598672,
                        0.5403023058681398,
                        -0.4161468365471424,
                        -0.9899924966004454,
                    ]
                ],
            ],
        ),
    )


def test_cos_sin_numerical_encoder_forward_2_features() -> None:
    module = CosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]),
        phase_shift=torch.tensor([[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]]),
    )
    assert objects_are_allclose(
        module(torch.tensor([[-1, -2], [0, 0], [1, 2]])),
        torch.tensor(
            [
                [
                    [
                        0.0,
                        -0.9092974066734314,
                        0.756802499294281,
                        1.0,
                        -0.416146844625473,
                        -0.6536436080932617,
                    ],
                    [
                        0.9589242935180664,
                        -0.9893582463264465,
                        0.9999902248382568,
                        0.28366219997406006,
                        -0.1455000340938568,
                        0.004425697959959507,
                    ],
                ],
                [
                    [
                        0.8414709568023682,
                        0.0,
                        -0.8414709568023682,
                        0.5403023362159729,
                        1.0,
                        0.5403023362159729,
                    ],
                    [
                        -0.8414709568023682,
                        0.0,
                        0.8414709568023682,
                        0.5403023362159729,
                        1.0,
                        0.5403023362159729,
                    ],
                ],
                [
                    [
                        0.9092974066734314,
                        0.9092974066734314,
                        0.9092974066734314,
                        -0.416146844625473,
                        -0.416146844625473,
                        -0.416146844625473,
                    ],
                    [
                        0.14112000167369843,
                        0.9893582463264465,
                        0.4201670289039612,
                        -0.9899924993515015,
                        -0.1455000340938568,
                        0.9074468016624451,
                    ],
                ],
            ]
        ),
    )


def test_cos_sin_numerical_encoder_forward_2_features_same() -> None:
    module = CosSinNumericalEncoder(
        frequency=torch.tensor([[1.0, 2.0, 3.0]]),
        phase_shift=torch.tensor([[0.0, 0.0, 0.0]]),
    )
    assert objects_are_allclose(
        module(torch.tensor([[-1, -2], [0, 0], [1, 2]])),
        torch.tensor(
            [
                [
                    [
                        -0.8414709568023682,
                        -0.9092974066734314,
                        -0.14112000167369843,
                        0.5403023362159729,
                        -0.416146844625473,
                        -0.9899924993515015,
                    ],
                    [
                        -0.9092974066734314,
                        0.756802499294281,
                        0.279415488243103,
                        -0.416146844625473,
                        -0.6536436080932617,
                        0.9601702690124512,
                    ],
                ],
                [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
                [
                    [
                        0.8414709568023682,
                        0.9092974066734314,
                        0.14112000167369843,
                        0.5403023362159729,
                        -0.416146844625473,
                        -0.9899924993515015,
                    ],
                    [
                        0.9092974066734314,
                        -0.756802499294281,
                        -0.279415488243103,
                        -0.416146844625473,
                        -0.6536436080932617,
                        0.9601702690124512,
                    ],
                ],
            ]
        ),
    )


##########################################
#     Tests for prepare_tensor_param     #
##########################################


@pytest.mark.parametrize("tensor", [torch.tensor([1.0, 2.0, 4.0]), torch.tensor([[1.0, 2.0, 4.0]])])
def test_prepare_tensor_param_1d(tensor: torch.Tensor) -> None:
    assert objects_are_equal(
        prepare_tensor_param(tensor, name="scale"), torch.tensor([[1.0, 2.0, 4.0]])
    )


def test_prepare_tensor_param_2d() -> None:
    assert objects_are_equal(
        prepare_tensor_param(torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]), name="scale"),
        torch.tensor([[1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]),
    )


def test_prepare_tensor_param_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="Incorrect shape for 'scale':"):
        prepare_tensor_param(torch.ones(2, 3, 4), name="scale")
