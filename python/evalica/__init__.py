from __future__ import annotations

from collections.abc import Collection, Hashable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

try:
    from numpy.exceptions import AxisError
except ImportError:
    from numpy import AxisError

from .evalica import (
    LengthMismatchError,
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
from .naive import counting as counting_naive
from .naive import eigen as eigen_naive
from .naive import elo as elo_naive
from .naive import newman as newman_naive
from .naive import pagerank as pagerank_naive

WINNERS = [
    Winner.X,
    Winner.Y,
    Winner.Draw,
    Winner.Ignore,
]

T = TypeVar("T", bound=Hashable)


def index_elements(
        xs: Collection[T],
        ys: Collection[T],
        index: pd.Index[T] | None = None,  # type: ignore[type-var]
) -> tuple[pd.Index[T], list[int], list[int]]:  # type: ignore[type-var]
    xy_index: dict[T, int] = {}

    def get_dict_index(x: T) -> int:
        if (idx := xy_index.get(x)) is None:
            idx = xy_index[x] = len(xy_index)

        return idx

    def get_pandas_index(x: T) -> int:
        return cast(int, cast("pd.Index[T]", index).get_loc(x))  # type: ignore[type-var]

    get_index = get_dict_index if index is None else get_pandas_index

    xs_indexed = [get_index(x) for x in xs]
    ys_indexed = [get_index(y) for y in ys]

    if index is None:
        index = pd.Index(xy_index)

    assert index is not None, "index is None"

    return index, xs_indexed, ys_indexed


@dataclass(frozen=True)
class MatricesResult(Generic[T]):
    win_matrix: npt.NDArray[np.int64]
    tie_matrix: npt.NDArray[np.int64]
    index: pd.Index[T]  # type: ignore[type-var]


def matrices(
        xs: Collection[T],
        ys: Collection[T],
        ws: Collection[Winner],
        index: pd.Index[T] | None = None,  # type: ignore[type-var]
) -> MatricesResult[T]:
    index, xs_indexed, ys_indexed = index_elements(xs, ys, index)

    assert index is not None, "index is None"

    win_matrix, tie_matrix = matrices_pyo3(xs_indexed, ys_indexed, ws)

    return MatricesResult(
        win_matrix=win_matrix,
        tie_matrix=tie_matrix,
        index=index,
    )


@dataclass(frozen=True)
class CountingResult(Generic[T]):
    scores: pd.Series[T]  # type: ignore[type-var]
    index: pd.Index[T]  # type: ignore[type-var]
    win_weight: float
    tie_weight: float
    solver: str


def counting(
        xs: Collection[T],
        ys: Collection[T],
        ws: Collection[Winner],
        index: pd.Index[T] | None = None,  # type: ignore[type-var]
        win_weight: float = 1.,
        tie_weight: float = .5,
        solver: Literal["naive", "pyo3"] = "pyo3",
) -> CountingResult[T]:
    index, xs_indexed, ys_indexed = index_elements(xs, ys, index)

    assert index is not None, "index is None"

    if solver == "pyo3":
        scores = counting_pyo3(xs_indexed, ys_indexed, ws, win_weight, tie_weight)
    else:
        scores = counting_naive(xs_indexed, ys_indexed, ws, win_weight, tie_weight)

    return CountingResult(
        scores=pd.Series(scores, index=index, name=counting.__name__),
        index=index,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver=solver,
    )


@dataclass(frozen=True)
class BradleyTerryResult(Generic[T]):
    scores: pd.Series[T]  # type: ignore[type-var]
    matrix: npt.NDArray[np.float64]
    index: pd.Index[T]  # type: ignore[type-var]
    win_weight: float
    tie_weight: float
    solver: str
    tolerance: float
    iterations: int


def bradley_terry(
        xs: Collection[T],
        ys: Collection[T],
        ws: Collection[Winner],
        index: pd.Index[T] | None = None,  # type: ignore[type-var]
        win_weight: float = 1.,
        tie_weight: float = .5,
        solver: Literal["naive", "pyo3"] = "pyo3",
        tolerance: float = 1e-6,
        limit: int = 100,
) -> BradleyTerryResult[T]:
    assert np.isfinite(win_weight), "win_weight must be finite"
    assert np.isfinite(tie_weight), "tie_weight must be finite"

    _matrices = matrices(xs, ys, ws, index)

    matrix = (win_weight * _matrices.win_matrix + tie_weight * _matrices.tie_matrix).astype(float)

    if solver == "pyo3":
        scores, iterations = bradley_terry_pyo3(matrix, tolerance, limit)
    else:
        scores, iterations = bradley_terry_naive(matrix, tolerance, limit)

    return BradleyTerryResult(
        scores=pd.Series(scores, index=_matrices.index, name=bradley_terry.__name__),
        matrix=matrix,
        index=_matrices.index,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver=solver,
        tolerance=tolerance,
        iterations=iterations,
    )


@dataclass(frozen=True)
class NewmanResult(Generic[T]):
    scores: pd.Series[T]  # type: ignore[type-var]
    win_matrix: npt.NDArray[np.float64]
    tie_matrix: npt.NDArray[np.float64]
    index: pd.Index[T]  # type: ignore[type-var]
    v: float
    v_init: float
    solver: str
    tolerance: float
    iterations: int


def newman(
        xs: Collection[T],
        ys: Collection[T],
        ws: Collection[Winner],
        index: pd.Index[T] | None = None,  # type: ignore[type-var]
        v_init: float = .5,
        solver: Literal["naive", "pyo3"] = "pyo3",
        tolerance: float = 1e-6,
        limit: int = 100,
) -> NewmanResult[T]:
    _matrices = matrices(xs, ys, ws, index)

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
        index=_matrices.index,
        v=v,
        v_init=v_init,
        solver=solver,
        tolerance=tolerance,
        iterations=iterations,
    )


@dataclass(frozen=True)
class EloResult(Generic[T]):
    scores: pd.Series[T]  # type: ignore[type-var]
    index: pd.Index[T]  # type: ignore[type-var]
    initial: float
    base: float
    scale: float
    k: float
    solver: str


def elo(
        xs: Collection[T],
        ys: Collection[T],
        ws: Collection[Winner],
        index: pd.Index[T] | None = None,  # type: ignore[type-var]
        initial: float = 1000.,
        base: float = 10.,
        scale: float = 400.,
        k: float = 30.,
        solver: Literal["naive", "pyo3"] = "pyo3",
) -> EloResult[T]:
    index, xs_indexed, ys_indexed = index_elements(xs, ys, index)

    assert index is not None, "index is None"

    if solver == "pyo3":
        scores = elo_pyo3(xs_indexed, ys_indexed, ws, initial, base, scale, k)
    else:
        scores = elo_naive(xs_indexed, ys_indexed, ws, initial, base, scale, k)

    return EloResult(
        scores=pd.Series(scores, index=index, name=elo.__name__),
        index=index,
        initial=initial,
        base=base,
        scale=scale,
        k=k,
        solver=solver,
    )


@dataclass(frozen=True)
class EigenResult(Generic[T]):
    scores: pd.Series[T]  # type: ignore[type-var]
    matrix: npt.NDArray[np.float64]
    index: pd.Index[T]  # type: ignore[type-var]
    win_weight: float
    tie_weight: float
    solver: str
    tolerance: float
    iterations: int


def eigen(
        xs: Collection[T],
        ys: Collection[T],
        ws: Collection[Winner],
        index: pd.Index[T] | None = None,  # type: ignore[type-var]
        win_weight: float = 1.,
        tie_weight: float = .5,
        solver: Literal["naive", "pyo3"] = "pyo3",
        tolerance: float = 1e-6,
        limit: int = 100,
) -> EigenResult[T]:
    assert np.isfinite(win_weight), "win_weight must be finite"
    assert np.isfinite(tie_weight), "tie_weight must be finite"

    _matrices = matrices(xs, ys, ws, index)

    matrix = (win_weight * _matrices.win_matrix + tie_weight * _matrices.tie_matrix).astype(float)

    if solver == "pyo3":
        scores, iterations = eigen_pyo3(matrix, tolerance, limit)
    else:
        scores, iterations = eigen_naive(matrix, tolerance, limit)

    return EigenResult(
        scores=pd.Series(scores, index=_matrices.index, name=eigen.__name__),
        matrix=matrix,
        index=_matrices.index,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver=solver,
        tolerance=tolerance,
        iterations=iterations,
    )


@dataclass(frozen=True)
class PageRankResult(Generic[T]):
    scores: pd.Series[T]  # type: ignore[type-var]
    index: pd.Index[T]  # type: ignore[type-var]
    damping: float
    win_weight: float
    tie_weight: float
    tolerance: float
    iterations: int


def pagerank(
        xs: Collection[T],
        ys: Collection[T],
        ws: Collection[Winner],
        index: pd.Index[T] | None = None,  # type: ignore[type-var]
        damping: float = .85,
        win_weight: float = 1.,
        tie_weight: float = .5,
        solver: Literal["naive", "pyo3"] = "pyo3",
        tolerance: float = 1e-6,
        limit: int = 100,
) -> PageRankResult[T]:
    _matrices = matrices(xs, ys, ws, index)

    win_matrix, tie_matrix = _matrices.win_matrix.astype(float), _matrices.tie_matrix.astype(float)

    if solver == "pyo3":
        scores, iterations = pagerank_pyo3(win_matrix, tie_matrix, damping, win_weight, tie_weight, tolerance, limit)
    else:
        scores, iterations = pagerank_naive(win_matrix, tie_matrix, damping, win_weight, tie_weight, tolerance, limit)

    return PageRankResult(
        scores=pd.Series(scores, index=_matrices.index, name=pagerank.__name__),
        index=_matrices.index,
        damping=damping,
        win_weight=win_weight,
        tie_weight=tie_weight,
        tolerance=tolerance,
        iterations=iterations,
    )


def pairwise_scores(scores: npt.NDArray[np.float64 | np.int64]) -> npt.NDArray[np.float64]:
    if scores.ndim != 1:
        raise AxisError(scores.ndim, 1)  # noqa: NPY201

    if not scores.shape[0]:
        return np.zeros((0, 0), dtype=np.float64)

    pairwise = scores[:, np.newaxis] / (scores + scores[:, np.newaxis])

    if np.isfinite(scores).all():
        pairwise = np.nan_to_num(pairwise)

    return pairwise


def pairwise_frame(scores: pd.Series[T]) -> pd.DataFrame:  # type: ignore[type-var]
    return pd.DataFrame(pairwise_scores(scores.to_numpy()), index=scores.index, columns=scores.index)


__all__ = [
    "BradleyTerryResult",
    "CountingResult",
    "EigenResult",
    "EloResult",
    "LengthMismatchError",
    "NewmanResult",
    "PageRankResult",
    "WINNERS",
    "Winner",
    "__version__",
    "bradley_terry",
    "counting",
    "eigen",
    "elo",
    "index_elements",
    "matrices",
    "newman",
    "pagerank",
    "pairwise_frame",
    "pairwise_scores",
]
