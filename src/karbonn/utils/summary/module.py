r"""Contain functionalities to analyze a ``torch.nn.Module``."""

from __future__ import annotations

__all__ = ["ModuleSummary", "parse_batch_dtype", "parse_batch_shape"]

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, overload

import torch
from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping
from torch import nn

from karbonn.utils.params import num_learnable_parameters, num_parameters

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

PARAMETER_NUM_UNITS = (" ", "K", "M", "B", "T")
UNKNOWN_SIZE = "?"
UNKNOWN_DTYPE = "?"


class ModuleSummary:
    r"""Summary class for a single layer in a ``torch.nn.Module``.

    It collects the following information:

    - Type of the layer (e.g. Linear, BatchNorm1d, ...)
    - Input shape
    - Output shape
    - Input data type
    - Output data type
    - Number of parameters
    - Number of learnable parameters

    The input and output shapes are only known after the example input
    array was passed through the model.

    Args:
        module: A module to summarize.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.utils.summary import ModuleSummary
    >>> model = torch.nn.Conv2d(3, 8, 3)
    >>> summary = ModuleSummary(model)
    >>> summary.get_num_parameters()
    224
    >>> summary.get_num_learnable_parameters()
    224
    >>> summary.get_layer_type()
    'Conv2d'
    >>> output = model(torch.rand(1, 3, 5, 5))
    >>> summary.get_in_size()
    torch.Size([1, 3, 5, 5])
    >>> summary.get_out_size()
    torch.Size([1, 8, 3, 3])
    >>> summary.get_in_dtype()
    torch.float32
    >>> summary.get_out_dtype()
    torch.float32
    >>> summary
    ModuleSummary(
      (module): Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1))
      (in_size): torch.Size([1, 3, 5, 5])
      (out_size): torch.Size([1, 8, 3, 3])
      (in_dtype): torch.float32
      (out_dtype): torch.float32
    )

    ```
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._module = module
        self._hook_handle = self._register_hook()
        self._in_size = None
        self._out_size = None
        self._in_dtype = None
        self._out_dtype = None

    def __del__(self) -> None:
        self.detach_hook()

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping(
                {
                    "module": self._module,
                    "in_size": self._in_size,
                    "out_size": self._out_size,
                    "in_dtype": self._in_dtype,
                    "out_dtype": self._out_dtype,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "module": self._module,
                    "in_size": self._in_size,
                    "out_size": self._out_size,
                    "in_dtype": self._in_dtype,
                    "out_dtype": self._out_dtype,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def _register_hook(self) -> RemovableHandle:
        r"""Register a hook on the module that computes the input and
        output size(s) on the first forward pass.

        If the hook is called, it will remove itself from the module,
        meaning that recursive models will only record their input and
        output shapes once.

        Return:
            A handle for the installed hook.
        """

        def hook(module: nn.Module, inp: Any, out: Any) -> None:  # noqa: ARG001
            if len(inp) == 1:
                inp = inp[0]
            self._in_size = parse_batch_shape(inp)
            self._out_size = parse_batch_shape(out)
            self._in_dtype = parse_batch_dtype(inp)
            self._out_dtype = parse_batch_dtype(out)
            self._hook_handle.remove()

        return self._module.register_forward_hook(hook)

    def detach_hook(self) -> None:
        r"""Remove the forward hook if it was not already removed in the
        forward pass.

        Will be called after the summary is created.
        """
        self._hook_handle.remove()

    def get_in_dtype(
        self,
    ) -> tuple[torch.dtype | None, ...] | dict[str, torch.dtype | None] | torch.dtype | None:
        r"""Return the input tensors data type.

        Returns:
            The input tensors data type.
        """
        return self._in_dtype

    def get_out_dtype(
        self,
    ) -> tuple[torch.dtype | None, ...] | dict[str, torch.dtype | None] | torch.dtype | None:
        r"""Return the output tensors data type.

        Returns:
            The output tensors data type.
        """
        return self._out_dtype

    def get_in_size(
        self,
    ) -> tuple[torch.Size | None, ...] | dict[str, torch.Size | None] | torch.Size | None:
        r"""Return the input tensors shapes.

        Returns:
            The input tensors shapes.
        """
        return self._in_size

    def get_out_size(
        self,
    ) -> tuple[torch.Size | None, ...] | dict[str, torch.Size | None] | torch.Size | None:
        r"""Return the output tensors shapes.

        Returns:
            The output tensors shapes.
        """
        return self._out_size

    def get_layer_type(self) -> str:
        r"""Return the class name of the module.

        Returns:
            The class name of the module.
        """
        return str(self._module.__class__.__qualname__)

    def get_num_parameters(self) -> int:
        r"""Return the number of parameters in this module.

        Returns:
            The number of parameters.
        """
        return num_parameters(self._module)

    def get_num_learnable_parameters(self) -> int:
        r"""Return the number of learnable parameters in this module.

        Returns:
            The number of learnable parameters.
        """
        return num_learnable_parameters(self._module)


@overload
def parse_batch_dtype(batch: torch.Tensor) -> torch.dtype | None: ...  # pragma: no cover


@overload
def parse_batch_dtype(
    batch: Sequence[torch.Tensor],
) -> tuple[torch.dtype | None, ...]: ...  # pragma: no cover


@overload
def parse_batch_dtype(
    batch: Mapping[str, torch.Tensor]
) -> dict[str, torch.dtype | None]: ...  # pragma: no cover


def parse_batch_dtype(
    batch: Any,
) -> tuple[torch.dtype | None, ...] | dict[str, torch.dtype | None] | torch.dtype | None:
    r"""Parse the data type of the tensors in the batch.

    The current implementation only parses the data type of a tensor,
    list of tensors, and dictionary of tensors.

    Args:
        batch: The batch to parse.

    Returns:
        The data types in the batch or ``None`` if it cannot parse the input.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.utils.summary.module import parse_batch_dtype
    >>> parse_batch_dtype(torch.ones(2, 3))
    torch.float32
    >>> parse_batch_dtype([torch.ones(2, 3), torch.zeros(2, dtype=torch.long)])
    (torch.float32, torch.int64)
    >>> parse_batch_dtype(
    ...     {"input1": torch.ones(2, 3), "input2": torch.zeros(2, dtype=torch.long)}
    ... )
    {'input1': torch.float32, 'input2': torch.int64}

    ```
    """
    if torch.is_tensor(batch):
        return batch.dtype
    if isinstance(batch, Sequence):
        return tuple(parse_batch_dtype(item) for item in batch)
    if isinstance(batch, Mapping):
        return {key: parse_batch_dtype(value) for key, value in batch.items()}
    return None


@overload
def parse_batch_shape(batch: torch.Tensor) -> torch.Size | None: ...  # pragma: no cover


@overload
def parse_batch_shape(
    batch: Sequence[torch.Tensor],
) -> tuple[torch.Size | None, ...]: ...  # pragma: no cover


@overload
def parse_batch_shape(
    batch: Mapping[str, torch.Tensor]
) -> dict[str, torch.Size | None]: ...  # pragma: no cover


def parse_batch_shape(
    batch: Any,
) -> tuple[torch.Size | None, ...] | dict[str, torch.Size | None] | torch.Size | None:
    r"""Parse the shapes of the tensors in the batch.

    The current implementation only parses the shapes of  tensor,
    list of tensors, and dictionary of tensors.

    Args:
        batch: The batch to parse.

    Returns:
        The shapes in the batch or ``None`` if it cannot parse the input.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.utils.summary.module import parse_batch_shape
    >>> parse_batch_shape(torch.ones(2, 3))
    torch.Size([2, 3])
    >>> parse_batch_shape([torch.ones(2, 3), torch.zeros(2)])
    (torch.Size([2, 3]), torch.Size([2]))
    >>> parse_batch_shape({"input1": torch.ones(2, 3), "input2": torch.zeros(2)})
    {'input1': torch.Size([2, 3]), 'input2': torch.Size([2])}

    ```
    """
    if torch.is_tensor(batch):
        return batch.shape
    if isinstance(batch, Sequence):
        return tuple(parse_batch_shape(item) for item in batch)
    if isinstance(batch, Mapping):
        return {key: parse_batch_shape(value) for key, value in batch.items()}
    return None