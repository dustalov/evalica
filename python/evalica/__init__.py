from collections import OrderedDict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd

from .evalica import (
    Winner,
    __version__,
    bradley_terry_pyo3,
    counting_pyo3,
    eigen_pyo3,
    elo_pyo3,
    matrices_pyo3,
    newman_pyo3,
)
from .naive import bradley_terry as bradley_terry_naive  # noqa: F401
from .naive import newman as newman_naive  # noqa: F401

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

def _index_elements(xs: Iterable[T], ys: Iterable[T]) -> tuple["pd.Index[T]", list[int], list[int]]:  # type: ignore[type-var]
    xy_index = index(xs, ys)

    xs_indexed = [xy_index[x] for x in xs]
    ys_indexed = [xy_index[y] for y in ys]

    return pd.Index(xy_index), xs_indexed, ys_indexed


@dataclass(frozen=True)
class MatricesResult(Generic[T]):
    win_matrix: npt.NDArray[np.int64]
    tie_matrix: npt.NDArray[np.int64]
    index: "pd.Index[T]"  # type: ignore[type-var]

def matrices(
    xs: Iterable[T],
    ys: Iterable[T],
    ws: Iterable[Winner],
) -> MatricesResult[T]:
    xy_index, _xs, _ys = _index_elements(xs, ys)

    W, T = matrices_pyo3(_xs, _ys, ws)  # noqa: N806

    return MatricesResult(
        win_matrix=W,
        tie_matrix=T,
        index=xy_index,
    )


@dataclass(frozen=True)
class CountingResult(Generic[T]):
    scores: "pd.Series[T]"  # type: ignore[type-var]
    win_matrix: npt.NDArray[np.int64]


def counting(
    xs: Iterable[T],
    ys: Iterable[T],
    ws: Iterable[Winner],
) -> CountingResult[T]:
    xy_index, _xs, _ys = _index_elements(xs, ys)

    W, _ = matrices_pyo3(_xs, _ys, ws)  # noqa: N806

    counts = counting_pyo3(W)

    return CountingResult(
        scores=pd.Series(counts, index=xy_index, name=counting.__name__),
        win_matrix=W,
    )

@dataclass(frozen=True)
class BradleyTerryResult(Generic[T]):
    scores: "pd.Series[T]"  # type: ignore[type-var]
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
) -> BradleyTerryResult[T]:
    assert np.isfinite(tie_weight), "tie_weight must be finite"

    xy_index, _xs, _ys = _index_elements(xs, ys)

    W, T = matrices_pyo3(_xs, _ys, ws)  # noqa: N806

    M = W.astype(float) + tie_weight * T.astype(float)  # noqa: N806

    scores, iterations = bradley_terry_pyo3(M, tolerance, limit)

    return BradleyTerryResult(
        scores=pd.Series(scores, index=xy_index, name=bradley_terry.__name__),
        matrix=M,
        tie_weight=tie_weight,
        iterations=iterations,
    )

@dataclass(frozen=True)
class NewmanResult(Generic[T]):
    scores: "pd.Series[T]"  # type: ignore[type-var]
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
) -> NewmanResult[T]:
    assert np.isfinite(v_init), "v_init must be finite"

    xy_index, _xs, _ys = _index_elements(xs, ys)

    W, T = matrices_pyo3(_xs, _ys, ws)  # noqa: N806
    W_float, T_float = W.astype(float), T.astype(float)  # noqa: N806

    scores, v, iterations = newman_pyo3(W_float, T_float, v_init, tolerance, limit)

    return NewmanResult(
        scores=pd.Series(scores, index=xy_index, name=newman.__name__),
        win_matrix=W_float,
        tie_matrix=T_float,
        v=v,
        v_init=v_init,
        iterations=iterations,
    )

@dataclass(frozen=True)
class EloResult(Generic[T]):
    scores: "pd.Series[T]"  # type: ignore[type-var]
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
) -> EloResult[T]:
    xy_index, _xs, _ys = _index_elements(xs, ys)

    scores = elo_pyo3(_xs, _ys, ws, r, k, s)

    return EloResult(
        scores=pd.Series(scores, index=xy_index, name=elo.__name__),
        r=r,
        k=k,
        s=s,
    )

@dataclass(frozen=True)
class EigenResult(Generic[T]):
    scores: "pd.Series[T]"  # type: ignore[type-var]
    matrix: npt.NDArray[np.float64]
    tie_weight: float

def eigen(
    xs: Iterable[T],
    ys: Iterable[T],
    ws: Iterable[Winner],
    tie_weight: float = .5,
) -> EigenResult[T]:
    xy_index, _xs, _ys = _index_elements(xs, ys)

    W, T = matrices_pyo3(_xs, _ys, ws)  # noqa: N806

    M = W.astype(float) + tie_weight * T.astype(float)  # noqa: N806

    scores = eigen_pyo3(M)

    return EigenResult(
        scores=pd.Series(scores, index=xy_index, name=eigen.__name__),
        matrix=M,
        tie_weight=tie_weight,
    )


def _pairwise_ndarray(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return scores[:, np.newaxis] / (scores + scores[:, np.newaxis])


def pairwise(scores: "pd.Series[T]") -> npt.NDArray[np.float64]:  # type: ignore[type-var]
    scores = scores.sort_values(ascending=False)
    return _pairwise_ndarray(scores.to_numpy())


__all__ = [
    "WINNERS",
    "Winner",
    "__version__",
    "bradley_terry",
    "counting",
    "eigen",
    "elo",
    "index",
    "matrices",
    "newman",
    "pairwise",
]
