from collections.abc import Collection
from enum import Enum

import numpy as np
import numpy.typing as npt

__version__: str = ...
"""The version of Evalica."""


class Winner(Enum):
    """The outcome of the pairwise comparison."""
    X = ...
    """The first element won."""
    Y = ...
    """The second element won."""
    Draw = ...
    """There is a tie."""
    Ignore = ...
    """The comparison should be ignored."""


class LengthMismatchError(ValueError):
    """The dataset dimensions mismatched."""


def matrices_pyo3(
        xs: npt.ArrayLike,
        ys: npt.ArrayLike,
        ws: Collection[Winner],
        total: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ...


def counting_pyo3(
        xs: npt.ArrayLike,
        ys: npt.ArrayLike,
        ws: Collection[Winner],
        total: int,
        win_weight: float,
        tie_weight: float,
) -> npt.NDArray[np.float64]: ...


def average_win_rate_pyo3(
        xs: npt.ArrayLike,
        ys: npt.ArrayLike,
        ws: Collection[Winner],
        total: int,
        win_weight: float,
        tie_weight: float,
) -> npt.NDArray[np.float64]: ...


def bradley_terry_pyo3(
        xs: npt.ArrayLike,
        ys: npt.ArrayLike,
        ws: Collection[Winner],
        total: int,
        win_weight: float,
        tie_weight: float,
        tolerance: float,
        limit: int,
) -> tuple[npt.NDArray[np.float64], int]: ...


def newman_pyo3(
        xs: npt.ArrayLike,
        ys: npt.ArrayLike,
        ws: Collection[Winner],
        total: int,
        v_init: float,
        win_weight: float,
        tie_weight: float,
        tolerance: float,
        limit: int,
) -> tuple[npt.NDArray[np.float64], float, int]: ...


def elo_pyo3(
        xs: npt.ArrayLike,
        ys: npt.ArrayLike,
        ws: Collection[Winner],
        total: int,
        initial: float,
        base: float,
        scale: float,
        k: float,
) -> npt.NDArray[np.float64]: ...


def eigen_pyo3(
        xs: npt.ArrayLike,
        ys: npt.ArrayLike,
        ws: Collection[Winner],
        total: int,
        win_weight: float,
        tie_weight: float,
        tolerance: float,
        limit: int,
) -> tuple[npt.NDArray[np.float64], int]: ...


def pagerank_pyo3(
        xs: npt.ArrayLike,
        ys: npt.ArrayLike,
        ws: Collection[Winner],
        total: int,
        damping: float,
        win_weight: float,
        tie_weight: float,
        tolerance: float,
        limit: int,
) -> tuple[npt.NDArray[np.float64], int]: ...
