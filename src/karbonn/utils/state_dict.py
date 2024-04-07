r"""Contain utility functions to manipulate ``torch.nn.Module``'s state
dict."""

from __future__ import annotations

__all__ = ["find_module_state_dict"]



def find_module_state_dict(state_dict: dict | list | tuple | set, module_keys: set) -> dict:
    r"""Try to find automatically the part of the state dict related to a
    module.

    The user should specify the set of module's keys:
    ``set(module.state_dict().keys())``. This function assumes that
    the set of keys only exists at one location in the state dict.
    If the set of keys exists at several locations in the state dict,
    only the first one is returned.

    Args:
        state_dict: The state dict. This function is called recursively
            on this input to find the queried state dict.
        module_keys: The set of module keys.

    Returns:
        The part of the state dict related to a module if it is
            found, otherwise an empty dict.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.utils import find_module_state_dict
    >>> state = {
    ...     "model": {
    ...         "weight": 42,
    ...         "network": {
    ...             "weight": torch.ones(5, 4),
    ...             "bias": 2 * torch.ones(5),
    ...         },
    ...     }
    ... }
    >>> module = torch.nn.Linear(4, 5)
    >>> state_dict = find_module_state_dict(state, module_keys=set(module.state_dict().keys()))
    >>> state_dict
    {'weight': tensor([[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]]), 'bias': tensor([2., 2., 2., 2., 2.])}

    ```
    """
    if isinstance(state_dict, dict):
        if set(state_dict.keys()) == module_keys:
            return state_dict
        for value in state_dict.values():
            state_dict = find_module_state_dict(value, module_keys)
            if state_dict:
                return state_dict
    elif isinstance(state_dict, (list, tuple, set)):
        for value in state_dict:
            state_dict = find_module_state_dict(value, module_keys)
            if state_dict:
                return state_dict
    return {}
