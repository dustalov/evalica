from enum import Enum

import numpy as np
import numpy.typing as npt

__version__: str = ...


class Winner(Enum):
    X = ...
    Y = ...
    Draw = ...
    Ignore = ...


def py_matrices(
        xs: npt.ArrayLike, ys: npt.ArrayLike, rs: list[Winner],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ...


def py_counting(m: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]: ...


def py_bradley_terry(
        m: npt.NDArray[np.int64], tolerance: float, limit: int,
) -> tuple[npt.NDArray[np.float64], int]: ...


def py_newman(
        w: npt.NDArray[np.float64], t: npt.NDArray[np.float64], v_init: float, tolerance: float, limit: int,
) -> tuple[npt.NDArray[np.float64], float, int]: ...


def py_elo(
        xs: npt.ArrayLike, ys: npt.ArrayLike, rs: list[Winner],
        r: float, k: int, s: float,
) -> npt.NDArray[np.float64]: ...


def py_eigen(
        m: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]: ...
