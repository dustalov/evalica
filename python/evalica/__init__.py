from collections import OrderedDict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd

from .evalica import Winner, __version__, py_bradley_terry, py_counting, py_eigen, py_elo, py_matrices, py_newman
from .naive import bradley_terry as bradley_terry_naive
from .naive import newman as newman_naive

WINNERS = [
    Winner.X,
    Winner.Y,
    Winner.Draw,
    Winner.Ignore,
]

T = TypeVar("T", bound=Hashable)

def index(xs: Iterable[T], *yss: Iterable[T]) -> dict[T, int]:
    index: dict[T, int] = OrderedDict()

    for ys in (xs, *yss):
        for y in ys:
            index[y] = index.get(y, len(index))

    return index

def _index_elements(xs: Iterable[T], ys: Iterable[T]) -> tuple["pd.Index[T]", list[int], list[int]]:
    xy_index = index(xs, ys)

    xs_indexed = [xy_index[x] for x in xs]
    ys_indexed = [xy_index[y] for y in ys]

    return pd.Index(xy_index), xs_indexed, ys_indexed


@dataclass(frozen=True)
class MatricesResult:
    win_matrix: npt.NDArray[np.int64]
    tie_matrix: npt.NDArray[np.int64]
    index: "pd.Index[Hashable]"

def matrices(
    xs: Iterable[T],
    ys: Iterable[T],
    ws: Iterable[Winner],
) -> MatricesResult:
    xy_index, _xs, _ys = _index_elements(xs, ys)

    W, T = py_matrices(_xs, _ys, ws)  # noqa: N806

    return MatricesResult(
        win_matrix=W,
        tie_matrix=T,
        index=xy_index,
    )


@dataclass(frozen=True)
class CountingResult:
    scores: "pd.Series[T]"
    win_matrix: npt.NDArray[np.int64]


def counting(
    xs: Iterable[T],
    ys: Iterable[T],
    ws: Iterable[Winner],
) -> CountingResult:
    xy_index, _xs, _ys = _index_elements(xs, ys)

    W, _ = py_matrices(_xs, _ys, ws)  # noqa: N806

    counts = py_counting(W)

    return CountingResult(
        scores=pd.Series(counts, index=xy_index, name=counting.__name__),
        win_matrix=W,
    )

@dataclass(frozen=True)
class BradleyTerryResult:
    scores: "pd.Series[T]"
    matrix: npt.NDArray[np.float64]
    tie_weight: float
    iterations: int

def bradley_terry(
    xs: Iterable[T],
    ys: Iterable[T],
    ws: Iterable[Winner],
    tie_weight: float = .5,
    tolerance: float = 1e-4,
    limit: int = 100,
) -> BradleyTerryResult:
    assert np.isfinite(tie_weight), "tie_weight must be finite"

    xy_index, _xs, _ys = _index_elements(xs, ys)

    W, T = py_matrices(_xs, _ys, ws)  # noqa: N806

    M = W.astype(float) + tie_weight * T.astype(float)  # noqa: N806

    scores, iterations = py_bradley_terry(M, tolerance, limit)

    return BradleyTerryResult(
        scores=pd.Series(scores, index=xy_index, name=bradley_terry.__name__),
        matrix=M,
        tie_weight=tie_weight,
        iterations=iterations,
    )

@dataclass(frozen=True)
class NewmanResult:
    scores: "pd.Series[T]"
    win_matrix: npt.NDArray[np.float64]
    tie_matrix: npt.NDArray[np.float64]
    v: float
    v_init: float
    iterations: int

def newman(
    xs: Iterable[T],
    ys: Iterable[T],
    ws: Iterable[Winner],
    v_init: float = .5,
    tolerance: float = 1e-4,
    limit: int = 100,
) -> NewmanResult:
    assert np.isfinite(v_init), "v_init must be finite"

    xy_index, _xs, _ys = _index_elements(xs, ys)

    W, T = py_matrices(_xs, _ys, ws)  # noqa: N806
    W_float, T_float = W.astype(float), T.astype(float)  # noqa: N806

    scores, v, iterations = py_newman(W_float, T_float, v_init, tolerance, limit)

    return NewmanResult(
        scores=pd.Series(scores, index=xy_index, name=newman.__name__),
        win_matrix=W_float,
        tie_matrix=T_float,
        v=v,
        v_init=v_init,
        iterations=iterations,
    )

@dataclass(frozen=True)
class EloResult:
    scores: "pd.Series[T]"
    r: float
    k: int
    s: float

def elo(
    xs: Iterable[T],
    ys: Iterable[T],
    ws: Iterable[Winner],
    r: float = 1500,
    k: int = 30,
    s: float = 400,
) -> EloResult:
    xy_index, _xs, _ys = _index_elements(xs, ys)

    scores = py_elo(_xs, _ys, ws, r, k, s)

    return EloResult(
        scores=pd.Series(scores, index=xy_index, name=elo.__name__),
        r=r,
        k=k,
        s=s,
    )

@dataclass(frozen=True)
class EigenResult:
    scores: "pd.Series[T]"
    matrix: npt.NDArray[np.float64]
    tie_weight: float

def eigen(
    xs: Iterable[T],
    ys: Iterable[T],
    ws: Iterable[Winner],
    tie_weight: float = .5,
) -> EigenResult:
    xy_index, _xs, _ys = _index_elements(xs, ys)

    W, T = py_matrices(_xs, _ys, ws)  # noqa: N806

    M = W.astype(float) + tie_weight * T.astype(float)  # noqa: N806

    scores = py_eigen(M)

    return EigenResult(
        scores=pd.Series(scores, index=xy_index, name=eigen.__name__),
        matrix=M,
        tie_weight=tie_weight,
    )

def pairwise(scores: "pd.Series[T] | npt.NDArray[np.float64]") -> npt.NDArray[np.float64]:
    if isinstance(scores, pd.Series):
        return pairwise(scores.sort_values(ascending=False).to_numpy())

    return scores[:, np.newaxis] / (scores + scores[:, np.newaxis])

__all__ = [
    "Winner",
    "__version__",
    "bradley_terry",
    "counting",
    "eigen",
    "elo",
    "py_matrices",
    "newman",
    "bradley_terry_naive",
    "newman_naive",
    "WINNERS",
    "index",
    "pairwise",
]
