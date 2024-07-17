from collections.abc import Collection
from enum import Enum

import numpy as np
import numpy.typing as npt

__version__: str = ...


class Winner(Enum):
    X = ...
    Y = ...
    Draw = ...
    Ignore = ...


class LengthMismatchError(ValueError):
    ...


def matrices_pyo3(
        xs: npt.ArrayLike,
        ys: npt.ArrayLike,
        ws: Collection[Winner],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ...


def counting_pyo3(
        xs: npt.ArrayLike,
        ys: npt.ArrayLike,
        ws: Collection[Winner],
        win_weight: float,
        tie_weight: float,
) -> npt.NDArray[np.float64]: ...


def bradley_terry_pyo3(
        xs: npt.ArrayLike,
        ys: npt.ArrayLike,
        ws: Collection[Winner],
        win_weight: float,
        tie_weight: float,
        tolerance: float,
        limit: int,
) -> tuple[npt.NDArray[np.float64], int]: ...


def newman_pyo3(
        xs: npt.ArrayLike,
        ys: npt.ArrayLike,
        ws: Collection[Winner],
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
        initial: float,
        base: float,
        scale: float,
        k: float,
) -> npt.NDArray[np.float64]: ...


def eigen_pyo3(
        xs: npt.ArrayLike,
        ys: npt.ArrayLike,
        ws: Collection[Winner],
        win_weight: float,
        tie_weight: float,
        tolerance: float,
        limit: int,
) -> tuple[npt.NDArray[np.float64], int]: ...


def pagerank_pyo3(
        xs: npt.ArrayLike,
        ys: npt.ArrayLike,
        ws: Collection[Winner],
        damping: float,
        win_weight: float,
        tie_weight: float,
        tolerance: float,
        limit: int,
) -> tuple[npt.NDArray[np.float64], int]: ...
