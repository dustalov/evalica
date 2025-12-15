"""Pure Python definitions for Evalica types and constants."""

from __future__ import annotations

from enum import Enum

# Read version from package metadata
try:
    from importlib.metadata import version
    __version__ = version("evalica")
except Exception:  # noqa: BLE001
    __version__ = "unknown"  # Fallback when package metadata unavailable


class Winner(Enum):
    """The outcome of the pairwise comparison."""

    X = 0
    """The first element won."""
    Y = 1
    """The second element won."""
    Draw = 2
    """There is a tie."""

    def __hash__(self) -> int:
        """Return hash of the winner value."""
        return hash(self.value)


class LengthMismatchError(ValueError):
    """The dataset dimensions mismatched."""

    def __init__(self, message: str = "mismatching input shapes") -> None:
        """Create and return a new object.

        Args:
            message: The error message.

        """
        super().__init__(message)


__all__ = [
    "Winner",
    "LengthMismatchError",
    "__version__",
]
