from collections.abc import Collection

import numpy as np
import numpy.typing as npt

from . import DistanceName

__version__: str = ...
"""The version of Evalica Brzo module."""

HAS_BLAS: bool = ...
"""Whether BLAS support is enabled."""

def matrices(
    xs: Collection[int],
    ys: Collection[int],
    winners: Collection[int],
    weights: Collection[float],
    total: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def pairwise_scores(scores: npt.ArrayLike) -> npt.NDArray[np.float64]: ...
def counting(
    xs: Collection[int],
    ys: Collection[int],
    winners: Collection[int],
    weights: Collection[float],
    total: int,
    win_weight: float,
    tie_weight: float,
) -> npt.NDArray[np.float64]: ...
def average_win_rate(
    xs: Collection[int],
    ys: Collection[int],
    winners: Collection[int],
    weights: Collection[float],
    total: int,
    win_weight: float,
    tie_weight: float,
) -> npt.NDArray[np.float64]: ...
def bradley_terry(
    xs: Collection[int],
    ys: Collection[int],
    winners: Collection[int],
    weights: Collection[float],
    total: int,
    win_weight: float,
    tie_weight: float,
    tolerance: float,
    limit: int,
) -> tuple[npt.NDArray[np.float64], int]: ...
def newman(
    xs: Collection[int],
    ys: Collection[int],
    winners: Collection[int],
    weights: Collection[float],
    total: int,
    v_init: float,
    win_weight: float,
    tie_weight: float,
    tolerance: float,
    limit: int,
) -> tuple[npt.NDArray[np.float64], float, int]: ...
def elo(
    xs: Collection[int],
    ys: Collection[int],
    winners: Collection[int],
    weights: Collection[float],
    total: int,
    initial: float,
    base: float,
    scale: float,
    k: float,
    win_weight: float,
    tie_weight: float,
) -> npt.NDArray[np.float64]: ...
def eigen(
    xs: Collection[int],
    ys: Collection[int],
    winners: Collection[int],
    weights: Collection[float],
    total: int,
    win_weight: float,
    tie_weight: float,
    tolerance: float,
    limit: int,
) -> tuple[npt.NDArray[np.float64], int]: ...
def pagerank(
    xs: Collection[int],
    ys: Collection[int],
    winners: Collection[int],
    weights: Collection[float],
    total: int,
    damping: float,
    win_weight: float,
    tie_weight: float,
    tolerance: float,
    limit: int,
) -> tuple[npt.NDArray[np.float64], int]: ...
def alpha(
    codes: npt.NDArray[np.int64],
    unique_values: npt.NDArray[np.float64],
    distance: DistanceName | npt.NDArray[np.float64],
) -> tuple[float, float, float]: ...
def alpha_bootstrap(
    codes: npt.NDArray[np.int64],
    unique_values: npt.NDArray[np.float64],
    distance: DistanceName | npt.NDArray[np.float64],
    n_resamples: int,
    random_state: int | None = ...,
) -> tuple[float, float, float, npt.NDArray[np.float64]]: ...
