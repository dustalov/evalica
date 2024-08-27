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


class LengthMismatchError(ValueError):
    """The dataset dimensions mismatched."""


def matrices_pyo3(
        xs: Collection[int],
        ys: Collection[int],
        winners: Collection[Winner],
        weights: Collection[float],
        total: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...


def pairwise_scores_pyo3(scores: npt.ArrayLike) -> npt.NDArray[np.float64]: ...


def counting_pyo3(
        xs: Collection[int],
        ys: Collection[int],
        winners: Collection[Winner],
        weights: Collection[float],
        total: int,
        win_weight: float,
        tie_weight: float,
) -> npt.NDArray[np.float64]: ...


def average_win_rate_pyo3(
        xs: Collection[int],
        ys: Collection[int],
        winners: Collection[Winner],
        weights: Collection[float],
        total: int,
        win_weight: float,
        tie_weight: float,
) -> npt.NDArray[np.float64]: ...


def bradley_terry_pyo3(
        xs: Collection[int],
        ys: Collection[int],
        winners: Collection[Winner],
        weights: Collection[float],
        total: int,
        win_weight: float,
        tie_weight: float,
        tolerance: float,
        limit: int,
) -> tuple[npt.NDArray[np.float64], int]: ...


def newman_pyo3(
        xs: Collection[int],
        ys: Collection[int],
        winners: Collection[Winner],
        weights: Collection[float],
        total: int,
        v_init: float,
        win_weight: float,
        tie_weight: float,
        tolerance: float,
        limit: int,
) -> tuple[npt.NDArray[np.float64], float, int]: ...


def elo_pyo3(
        xs: Collection[int],
        ys: Collection[int],
        winners: Collection[Winner],
        weights: Collection[float],
        total: int,
        initial: float,
        base: float,
        scale: float,
        k: float,
        win_weight: float,
        tie_weight: float,
) -> npt.NDArray[np.float64]: ...


def eigen_pyo3(
        xs: Collection[int],
        ys: Collection[int],
        winners: Collection[Winner],
        weights: Collection[float],
        total: int,
        win_weight: float,
        tie_weight: float,
        tolerance: float,
        limit: int,
) -> tuple[npt.NDArray[np.float64], int]: ...


def pagerank_pyo3(
        xs: Collection[int],
        ys: Collection[int],
        winners: Collection[Winner],
        weights: Collection[float],
        total: int,
        damping: float,
        win_weight: float,
        tie_weight: float,
        tolerance: float,
        limit: int,
) -> tuple[npt.NDArray[np.float64], int]: ...
