from __future__ import annotations

from unittest.mock import patch

import torch

from karbonn.utils.tensor import FlattenBuffer

###################################
#     Tests for FlattenBuffer     #
###################################


def test_flatten_buffer_init_values_none() -> None:
    buffer = FlattenBuffer()
    assert buffer._values.equal(torch.tensor([]))
    assert not buffer._buffer


def test_flatten_buffer_init_values_tensor() -> None:
    buffer = FlattenBuffer(values=torch.arange(4))
    assert buffer._values.equal(torch.arange(4))
    assert not buffer._buffer


def test_flatten_buffer_str() -> None:
    assert str(FlattenBuffer()).startswith("FlattenBuffer(")


@patch("karbonn.utils.tensor.buffer.all_gather_tensor_varshape", lambda tensor: [tensor])
def test_flatten_buffer_all_reduce_non_distributed() -> None:
    buffer = FlattenBuffer(values=torch.arange(4))
    buffer.update(torch.tensor([-3, 1, 7]))
    buffer_reduced = buffer.all_reduce()
    assert buffer is not buffer_reduced
    assert buffer.equal(FlattenBuffer(torch.tensor([0, 1, 2, 3, -3, 1, 7], dtype=torch.long)))
    assert buffer_reduced.equal(
        FlattenBuffer(torch.tensor([0, 1, 2, 3, -3, 1, 7], dtype=torch.long))
    )


@patch(
    "karbonn.utils.tensor.buffer.all_gather_tensor_varshape",
    lambda tensor: [tensor, torch.tensor([3, 2, 1])],
)
def test_flatten_buffer_all_reduce_distributed() -> None:
    buffer = FlattenBuffer(values=torch.arange(4))
    buffer.update(torch.tensor([-3, 1, 7]))
    buffer_reduced = buffer.all_reduce()
    assert buffer is not buffer_reduced
    assert buffer.equal(FlattenBuffer(torch.tensor([0, 1, 2, 3, -3, 1, 7], dtype=torch.long)))
    assert buffer_reduced.equal(
        FlattenBuffer(torch.tensor([0, 1, 2, 3, -3, 1, 7, 3, 2, 1], dtype=torch.long))
    )


def test_flatten_buffer_all_reduce_empty() -> None:
    buffer = FlattenBuffer()
    buffer_reduced = buffer.all_reduce()
    assert buffer is not buffer_reduced
    assert buffer.equal(FlattenBuffer())
    assert buffer_reduced.equal(FlattenBuffer())


def test_flatten_buffer_clear() -> None:
    buffer = FlattenBuffer(values=torch.arange(4))
    buffer.update(torch.tensor([-3, 1, 7]))
    buffer.clear()
    assert buffer.equal(FlattenBuffer())


def test_flatten_buffer_clear_empty() -> None:
    buffer = FlattenBuffer()
    buffer.clear()
    assert buffer.equal(FlattenBuffer())


def test_flatten_buffer_clone_values_without_buffer() -> None:
    buffer = FlattenBuffer(torch.arange(6))
    buffer_cloned = buffer.clone()
    buffer.values().add_(1)
    assert buffer is not buffer_cloned
    assert buffer.equal(FlattenBuffer(torch.arange(6).add(1)))
    assert buffer_cloned.equal(FlattenBuffer(torch.arange(6)))


def test_flatten_buffer_clone_values_with_buffer() -> None:
    buffer = FlattenBuffer(torch.arange(6))
    buffer.update(torch.tensor([-3, 1, 7]))
    buffer_cloned = buffer.clone()
    buffer.values().add_(1)
    assert buffer is not buffer_cloned
    assert buffer.equal(FlattenBuffer(torch.tensor([1, 2, 3, 4, 5, 6, -2, 2, 8])))
    assert buffer_cloned.equal(FlattenBuffer(torch.tensor([0, 1, 2, 3, 4, 5, -3, 1, 7])))


def test_flatten_buffer_clone_empty() -> None:
    buffer = FlattenBuffer()
    buffer_cloned = buffer.clone()
    assert buffer is not buffer_cloned
    assert buffer.equal(FlattenBuffer())
    assert buffer_cloned.equal(FlattenBuffer())


def test_flatten_buffer_consolidate() -> None:
    buffer = FlattenBuffer(values=torch.arange(4))
    buffer.update(torch.tensor([-3, 1, 7]))
    buffer.consolidate()
    assert buffer.equal(FlattenBuffer(torch.tensor([0, 1, 2, 3, -3, 1, 7], dtype=torch.long)))


def test_flatten_buffer_consolidate_empty_buffer() -> None:
    buffer = FlattenBuffer(values=torch.arange(4))
    buffer.consolidate()
    assert buffer.equal(FlattenBuffer(torch.tensor([0, 1, 2, 3], dtype=torch.long)))


def test_flatten_buffer_consolidate_empty() -> None:
    buffer = FlattenBuffer()
    buffer.consolidate()
    assert buffer.equal(FlattenBuffer())


def test_flatten_buffer_equal_true_values_without_buffer() -> None:
    assert FlattenBuffer(torch.arange(6)).equal(FlattenBuffer(torch.arange(6)))


def test_flatten_buffer_equal_true_values_with_buffer() -> None:
    tensor1 = FlattenBuffer(torch.arange(6))
    tensor1.update(torch.tensor([-1.0, 4.0]))
    tensor2 = FlattenBuffer(torch.arange(6))
    tensor2.update(torch.tensor([-1.0, 4.0]))
    assert tensor1.equal(tensor2)


def test_flatten_buffer_equal_true_empty() -> None:
    assert FlattenBuffer().equal(FlattenBuffer())


def test_flatten_buffer_equal_false_different_type() -> None:
    assert not FlattenBuffer().equal(torch.arange(6))


def test_flatten_buffer_equal_false_self_empty() -> None:
    assert not FlattenBuffer().equal(FlattenBuffer(torch.arange(6)))


def test_flatten_buffer_equal_false_other_empty() -> None:
    assert not FlattenBuffer(torch.arange(6)).equal(FlattenBuffer())


def test_flatten_buffer_equal_false_same_values_different_buffers() -> None:
    tensor1 = FlattenBuffer(torch.arange(6))
    tensor1.update(torch.tensor([-1.0, 4.0]))
    tensor2 = FlattenBuffer(torch.arange(6))
    tensor2.update(torch.tensor([-2.0, 4.0]))
    assert not tensor1.equal(tensor2)


def test_flatten_buffer_numel_empty() -> None:
    assert FlattenBuffer().numel() == 0


def test_flatten_buffer_numel_without_buffer() -> None:
    assert FlattenBuffer(torch.arange(6)).numel() == 6


def test_flatten_buffer_numel_with_buffer() -> None:
    buffer = FlattenBuffer(torch.arange(6))
    buffer.update(torch.tensor([-3, 1, 7]))
    assert buffer.numel() == 9


def test_flatten_buffer_update_1d_tensor() -> None:
    buffer = FlattenBuffer()
    buffer.update(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float))
    assert buffer.values().equal(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float))


def test_flatten_buffer_update_2d_tensor() -> None:
    buffer = FlattenBuffer()
    buffer.update(torch.arange(6, dtype=torch.float).view(2, 3))
    assert buffer.values().equal(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float))


def test_flatten_buffer_update_float_tensor() -> None:
    buffer = FlattenBuffer()
    buffer.update(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float))
    assert buffer.values().equal(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float))


def test_flatten_buffer_update_long_tensor() -> None:
    buffer = FlattenBuffer()
    buffer.update(torch.tensor([-3, 1, 7]))
    assert buffer.values().equal(torch.tensor([-3, 1, 7], dtype=torch.long))


def test_flatten_buffer_values() -> None:
    buffer = FlattenBuffer(values=torch.arange(4))
    buffer.update(torch.tensor([-3, 1, 7]))
    assert buffer.values().equal(torch.tensor([0, 1, 2, 3, -3, 1, 7], dtype=torch.long))


def test_flatten_buffer_values_empty() -> None:
    assert FlattenBuffer().values().equal(torch.tensor([]))


def test_flatten_buffer_values_duplicate_call() -> None:
    buffer = FlattenBuffer(values=torch.arange(4))
    buffer.update(torch.tensor([-3, 1, 7]))
    values1 = buffer.values()
    assert values1.equal(torch.tensor([0, 1, 2, 3, -3, 1, 7], dtype=torch.long))
    values2 = buffer.values()
    assert values2.equal(torch.tensor([0, 1, 2, 3, -3, 1, 7], dtype=torch.long))
    assert values1 is values2
