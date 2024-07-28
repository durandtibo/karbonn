r"""Define the exceptions for states."""

from __future__ import annotations

__all__ = ["EmptyStateError"]


class EmptyStateError(Exception):
    r"""Raised if the state is empty because it is not possible to
    evaluate an empty state."""
