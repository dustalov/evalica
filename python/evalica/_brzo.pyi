from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from . import DistanceName

_Indices: TypeAlias = Sequence[int] | npt.NDArray[np.uintp]
_Winners: TypeAlias = Sequence[int] | npt.NDArray[np.uint8]
_Weights: TypeAlias = Sequence[float] | npt.NDArray[np.float64]

__version__: str = ...
"""The version of Evalica Brzo module."""

def indices(indices: Sequence[int], total: int) -> npt.NDArray[np.uintp]: ...
def matrices(
    xs: _Indices,
    ys: _Indices,
    winners: _Winners,
    weights: _Weights,
    total: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def pairwise_scores(scores: _Weights) -> npt.NDArray[np.float64]: ...
def counting(
    xs: _Indices,
    ys: _Indices,
    winners: _Winners,
    weights: _Weights,
    total: int,
    win_weight: float,
    tie_weight: float,
) -> npt.NDArray[np.float64]: ...
def average_win_rate(
    xs: _Indices,
    ys: _Indices,
    winners: _Winners,
    weights: _Weights,
    total: int,
    win_weight: float,
    tie_weight: float,
) -> npt.NDArray[np.float64]: ...
def bradley_terry(
    xs: _Indices,
    ys: _Indices,
    winners: _Winners,
    weights: _Weights,
    total: int,
    win_weight: float,
    tie_weight: float,
    tolerance: float,
    limit: int,
) -> tuple[npt.NDArray[np.float64], int]: ...
def newman(
    xs: _Indices,
    ys: _Indices,
    winners: _Winners,
    weights: _Weights,
    total: int,
    v_init: float,
    win_weight: float,
    tie_weight: float,
    tolerance: float,
    limit: int,
) -> tuple[npt.NDArray[np.float64], float, int]: ...
def elo(
    xs: _Indices,
    ys: _Indices,
    winners: _Winners,
    weights: _Weights,
    total: int,
    initial: float,
    base: float,
    scale: float,
    k: float,
    win_weight: float,
    tie_weight: float,
) -> npt.NDArray[np.float64]: ...
def eigen(
    xs: _Indices,
    ys: _Indices,
    winners: _Winners,
    weights: _Weights,
    total: int,
    win_weight: float,
    tie_weight: float,
    tolerance: float,
    limit: int,
) -> tuple[npt.NDArray[np.float64], int]: ...
def pagerank(
    xs: _Indices,
    ys: _Indices,
    winners: _Winners,
    weights: _Weights,
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
