import dataclasses
from collections import OrderedDict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

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
    pagerank_pyo3,
)
from .naive import bradley_terry as bradley_terry_naive
from .naive import eigen as eigen_naive
from .naive import newman as newman_naive

WINNERS = [
    Winner.X,
    Winner.Y,
    Winner.Draw,
    Winner.Ignore,
]

T = TypeVar("T", bound=Hashable)


def enumerate_elements(xs: Iterable[T], *yss: Iterable[T]) -> dict[T, int]:
    index: dict[T, int] = OrderedDict()

    for ys in (xs, *yss):
        for y in ys:
            index[y] = index.get(y, len(index))

    return index


@dataclass
class IndexedElements(Generic[T]):
    index: "pd.Index[T]"  # type: ignore[type-var]
    xs: list[int]
    ys: list[int]


def index_elements(xs: Iterable[T], ys: Iterable[T]) -> IndexedElements[T]:
    xy_index = enumerate_elements(xs, ys)

    xs_indexed = [xy_index[x] for x in xs]
    ys_indexed = [xy_index[y] for y in ys]

    return IndexedElements(
        index=pd.Index(xy_index),
        xs=xs_indexed,
        ys=ys_indexed,
    )


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
    index, _xs, _ys = dataclasses.astuple(index_elements(xs, ys))

    win_matrix, tie_matrix = matrices_pyo3(_xs, _ys, ws)

    return MatricesResult(
        win_matrix=win_matrix,
        tie_matrix=tie_matrix,
        index=index,
    )


@dataclass(frozen=True)
class CountingResult(Generic[T]):
    scores: "pd.Series[T]"  # type: ignore[type-var]
    win_weight: float
    tie_weight: float


def counting(
        xs: Iterable[T],
        ys: Iterable[T],
        ws: Iterable[Winner],
        win_weight: float = 1.,
        tie_weight: float = .5,
) -> CountingResult[T]:
    index, _xs, _ys = dataclasses.astuple(index_elements(xs, ys))

    counts = counting_pyo3(_xs, _ys, ws, win_weight, tie_weight)

    return CountingResult(
        scores=pd.Series(counts, index=index, name=counting.__name__),
        win_weight=win_weight,
        tie_weight=tie_weight,
    )


@dataclass(frozen=True)
class BradleyTerryResult(Generic[T]):
    scores: "pd.Series[T]"  # type: ignore[type-var]
    matrix: npt.NDArray[np.float64]
    win_weight: float
    tie_weight: float
    iterations: int


def bradley_terry(
        xs: Iterable[T],
        ys: Iterable[T],
        ws: Iterable[Winner],
        win_weight: float = 1.,
        tie_weight: float = .5,
        solver: Literal["naive", "pyo3"] = "pyo3",
        tolerance: float = 1e-4,
        limit: int = 100,
) -> BradleyTerryResult[T]:
    assert np.isfinite(tie_weight), "tie_weight must be finite"

    _matrices = matrices(xs, ys, ws)

    matrix = (win_weight * _matrices.win_matrix + tie_weight * _matrices.tie_matrix).astype(float)

    if solver == "pyo3":
        scores, iterations = bradley_terry_pyo3(matrix, tolerance, limit)
    else:
        scores, iterations = bradley_terry_naive(matrix, tolerance, limit)

    return BradleyTerryResult(
        scores=pd.Series(scores, index=_matrices.index, name=bradley_terry.__name__),
        matrix=matrix,
        win_weight=win_weight,
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
        solver: Literal["naive", "pyo3"] = "pyo3",
        tolerance: float = 1e-4,
        limit: int = 100,
) -> NewmanResult[T]:
    assert np.isfinite(v_init), "v_init must be finite"

    _matrices = matrices(xs, ys, ws)

    win_matrix = _matrices.win_matrix.astype(float)
    tie_matrix = _matrices.tie_matrix.astype(float)

    if solver == "pyo3":
        scores, v, iterations = newman_pyo3(win_matrix, tie_matrix, v_init, tolerance, limit)
    else:
        scores, v, iterations = newman_naive(win_matrix, tie_matrix, v_init, tolerance, limit)

    return NewmanResult(
        scores=pd.Series(scores, index=_matrices.index, name=newman.__name__),
        win_matrix=win_matrix,
        tie_matrix=tie_matrix,
        v=v,
        v_init=v_init,
        iterations=iterations,
    )


@dataclass(frozen=True)
class EloResult(Generic[T]):
    scores: "pd.Series[T]"  # type: ignore[type-var]
    initial: float
    base: float
    scale: float
    k: float


def elo(
        xs: Iterable[T],
        ys: Iterable[T],
        ws: Iterable[Winner],
        initial: float = 1500.,
        base: float = 10.,
        scale: float = 400.,
        k: float = 30.,
) -> EloResult[T]:
    index, _xs, _ys = dataclasses.astuple(index_elements(xs, ys))

    scores = elo_pyo3(_xs, _ys, ws, initial, base, scale, k)

    return EloResult(
        scores=pd.Series(scores, index=index, name=elo.__name__),
        initial=initial,
        base=base,
        scale=scale,
        k=k,
    )


@dataclass(frozen=True)
class EigenResult(Generic[T]):
    scores: "pd.Series[T]"  # type: ignore[type-var]
    matrix: npt.NDArray[np.float64]
    win_weight: float
    tie_weight: float


def eigen(
        xs: Iterable[T],
        ys: Iterable[T],
        ws: Iterable[Winner],
        win_weight: float = 1.,
        tie_weight: float = .5,
        solver: Literal["naive", "pyo3"] = "pyo3",
) -> EigenResult[T]:
    _matrices = matrices(xs, ys, ws)

    matrix = (win_weight * _matrices.win_matrix + tie_weight * _matrices.tie_matrix).astype(float)

    scores = eigen_pyo3(matrix) if solver == "pyo3" else eigen_naive(matrix)[0]

    return EigenResult(
        scores=pd.Series(scores, index=_matrices.index, name=eigen.__name__),
        matrix=matrix,
        win_weight=win_weight,
        tie_weight=tie_weight,
    )


@dataclass(frozen=True)
class PageRankResult(Generic[T]):
    scores: "pd.Series[T]"  # type: ignore[type-var]
    damping: float
    win_weight: float
    tie_weight: float
    iterations: int


def pagerank(
        xs: Iterable[T],
        ys: Iterable[T],
        ws: Iterable[Winner],
        damping: float = .85,
        win_weight: float = 1.,
        tie_weight: float = .5,
        tolerance: float = 1e-4,
        limit: int = 100,
) -> PageRankResult[T]:
    index, _xs, _ys = dataclasses.astuple(index_elements(xs, ys))

    scores, iterations = pagerank_pyo3(_xs, _ys, ws, damping, win_weight, tie_weight, tolerance, limit)

    return PageRankResult(
        scores=pd.Series(scores, index=index, name=pagerank.__name__),
        damping=damping,
        win_weight=win_weight,
        tie_weight=tie_weight,
        iterations=iterations,
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
    "enumerate_elements",
    "matrices",
    "newman",
    "pagerank",
    "pairwise",
]
