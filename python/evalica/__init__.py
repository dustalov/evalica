"""Evalica, your favourite evaluation suite."""

from __future__ import annotations

import warnings
from collections.abc import Collection, Hashable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, Literal, Protocol, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

from .evalica import (
    LengthMismatchError,
    Winner,
    __version__,
    average_win_rate_pyo3,
    bradley_terry_pyo3,
    counting_pyo3,
    eigen_pyo3,
    elo_pyo3,
    matrices_pyo3,
    newman_pyo3,
    pagerank_pyo3,
    pairwise_scores_pyo3,
)
from .naive import bradley_terry as bradley_terry_naive
from .naive import counting as counting_naive
from .naive import eigen as eigen_naive
from .naive import elo as elo_naive
from .naive import newman as newman_naive
from .naive import pagerank as pagerank_naive
from .naive import pairwise_scores as pairwise_scores_naive

WINNERS = [
    Winner.X,
    Winner.Y,
    Winner.Draw,
]
"""Known values of Winner."""

T = TypeVar("T", bound=Hashable)


def _wrap_weights(weights: Collection[float] | None, n: int) -> Collection[float]:
    if weights is None:
        return np.repeat(1., n)

    assert np.isfinite(weights).all(), "weights must be finite"  # type: ignore[call-overload]

    return weights


def _make_matrix(
        win_matrix: npt.NDArray[np.float64],
        tie_matrix: npt.NDArray[np.float64],
        win_weight: float = 1.,
        tie_weight: float = .5,
        nan: float = 0.0,
) -> npt.NDArray[np.float64]:
    with np.errstate(all="ignore"):
        return np.nan_to_num(
            win_weight * np.nan_to_num(win_matrix, nan=nan) +
            tie_weight * np.nan_to_num(tie_matrix, nan=nan),
            nan=nan,
        )


def indexing(
        xs: Collection[T],
        ys: Collection[T],
        index: dict[T, int] | None = None,
) -> tuple[list[int], list[int], dict[T, int]]:
    """
    Map the input elements into their numerical representations.

    Args:
        xs: The left-hand side elements.
        ys: The right-hand side elements.
        index: The pre-computed index.

    Returns:
        The tuple containing the numerical representations of the input elements and the corresponding index.

    """
    if index is None:
        index = {}
        xy_index = index
    else:
        xy_index = MappingProxyType(index)  # type: ignore[assignment]

    def get_index(x: T) -> int:
        if (idx := xy_index.get(x)) is None:
            idx = xy_index[x] = len(xy_index)

        return idx

    xs_indexed = [get_index(x) for x in xs]
    ys_indexed = [get_index(y) for y in ys]

    return xs_indexed, ys_indexed, index


@dataclass(frozen=True)
class MatricesResult(Generic[T]):
    """
    The win and tie matrices.

    Attributes:
        win_matrix: The matrix representing wins between the elements.
        tie_matrix: The matrix representing ties between the elements; it is always symmetric.
        index: The index.

    """

    win_matrix: npt.NDArray[np.float64]
    tie_matrix: npt.NDArray[np.float64]
    index: dict[T, int]


def matrices(
        xs_indexed: Collection[int],
        ys_indexed: Collection[int],
        winners: Collection[Winner],
        index: dict[T, int],
        weights: Collection[float] | None = None,
) -> MatricesResult[T]:
    """
    Build win and tie matrices from the given elements.

    Args:
        xs_indexed: The left-hand side elements.
        ys_indexed: The right-hand side elements.
        winners: The winner elements.
        index: The index.
        weights: The example weights.

    Returns:
        The win and tie matrices.

    """
    weights = _wrap_weights(weights, len(xs_indexed))

    win_matrix, tie_matrix = matrices_pyo3(
        xs=xs_indexed,
        ys=ys_indexed,
        winners=winners,
        weights=weights,
        total=len(index),
    )

    return MatricesResult(
        win_matrix=win_matrix,
        tie_matrix=tie_matrix,
        index=index,
    )


@runtime_checkable
class Result(Protocol[T]):
    """
    The result protocol.

    Attributes:
        scores: The element scores.
        index: The index.

    """

    scores: pd.Series[float]
    index: dict[T, int]


@dataclass(frozen=True)
class CountingResult(Generic[T]):
    """
    The counting result.

    Attributes:
        scores: The element scores.
        index: The index.
        win_weight: The win weight.
        tie_weight: The tie weight.
        solver: The solver.

    """

    scores: pd.Series[float]
    index: dict[T, int]
    win_weight: float
    tie_weight: float
    solver: str


def counting(
        xs: Collection[T],
        ys: Collection[T],
        winners: Collection[Winner],
        index: dict[T, int] | None = None,
        weights: Collection[float] | None = None,
        win_weight: float = 1.,
        tie_weight: float = .5,
        solver: Literal["naive", "pyo3"] = "pyo3",
) -> CountingResult[T]:
    """
    Count individual elements.

    Args:
        xs: The left-hand side elements.
        ys: The right-hand side elements.
        winners: The winner elements.
        index: The index.
        weights: The example weights.
        win_weight: The win weight.
        tie_weight: The tie weight.
        solver: The solver.

    Returns:
        The counting result.

    """
    assert np.isfinite(win_weight), "win_weight must be finite"
    assert np.isfinite(tie_weight), "tie_weight must be finite"

    xs_indexed, ys_indexed, index = indexing(xs, ys, index)

    assert index is not None, "index is None"

    weights = _wrap_weights(weights, len(xs_indexed))

    if solver == "pyo3":
        scores = counting_pyo3(
            xs=xs_indexed,
            ys=ys_indexed,
            winners=winners,
            weights=weights,
            total=len(index),
            win_weight=win_weight,
            tie_weight=tie_weight,
        )
    else:
        scores = counting_naive(
            xs=xs_indexed,
            ys=ys_indexed,
            winners=winners,
            weights=weights,
            total=len(index),
            win_weight=win_weight,
            tie_weight=tie_weight,
        )

    return CountingResult(
        scores=pd.Series(scores, index=index, name=counting.__name__).sort_values(ascending=False, kind="stable"),
        index=index,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver=solver,
    )


@dataclass(frozen=True)
class AverageWinRateResult(Generic[T]):
    """
    The average win rate result.

    Attributes:
        scores: The element scores.
        index: The index.
        win_weight: The win weight.
        tie_weight: The tie weight.
        solver: The solver.

    """

    scores: pd.Series[float]
    index: dict[T, int]
    win_weight: float
    tie_weight: float
    solver: str


def average_win_rate(
        xs: Collection[T],
        ys: Collection[T],
        winners: Collection[Winner],
        index: dict[T, int] | None = None,
        weights: Collection[float] | None = None,
        win_weight: float = 1.,
        tie_weight: float = .5,
        solver: Literal["naive", "pyo3"] = "pyo3",
) -> AverageWinRateResult[T]:
    """
    Count pairwise win rates between the elements and average per element.

    Args:
        xs: The left-hand side elements.
        ys: The right-hand side elements.
        winners: The winner elements.
        index: The index.
        weights: The example weights.
        win_weight: The win weight.
        tie_weight: The tie weight.
        solver: The solver.

    Returns:
        The average win rate result.

    """
    assert np.isfinite(win_weight), "win_weight must be finite"
    assert np.isfinite(tie_weight), "tie_weight must be finite"

    xs_indexed, ys_indexed, index = indexing(xs, ys, index)

    assert index is not None, "index is None"

    weights = _wrap_weights(weights, len(xs_indexed))

    if solver == "pyo3":
        scores = average_win_rate_pyo3(
            xs=xs_indexed,
            ys=ys_indexed,
            winners=winners,
            weights=weights,
            total=len(index),
            win_weight=win_weight,
            tie_weight=tie_weight,
        )

    else:
        _matrices = matrices(
            xs_indexed=xs_indexed,
            ys_indexed=ys_indexed,
            winners=winners,
            index=index,
            weights=weights,
        )

        matrix = _make_matrix(_matrices.win_matrix, _matrices.tie_matrix, win_weight, tie_weight)

        with np.errstate(all="ignore"):
            denominator = np.nan_to_num(matrix + matrix.T)

            matrix /= denominator

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Mean of empty slice")

            scores = np.nan_to_num(np.nanmean(matrix, axis=1), copy=False)

    return AverageWinRateResult(
        scores=pd.Series(scores, index=index, name=average_win_rate.__name__).sort_values(
            ascending=False, kind="stable"),
        index=index,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver=solver,
    )


@dataclass(frozen=True)
class BradleyTerryResult(Generic[T]):
    """
    The Bradley-Terry result.

    Attributes:
        scores: The element scores.
        index: The index.
        win_weight: The win weight.
        tie_weight: The tie weight.
        solver: The solver.
        tolerance: The convergence tolerance.
        iterations: The actual number of iterations.
        limit: The maximum number of iterations.

    """

    scores: pd.Series[float]
    index: dict[T, int]
    win_weight: float
    tie_weight: float
    solver: str
    tolerance: float
    iterations: int
    limit: int


def bradley_terry(
        xs: Collection[T],
        ys: Collection[T],
        winners: Collection[Winner],
        index: dict[T, int] | None = None,
        weights: Collection[float] | None = None,
        win_weight: float = 1.,
        tie_weight: float = .5,
        solver: Literal["naive", "pyo3"] = "pyo3",
        tolerance: float = 1e-6,
        limit: int = 100,
) -> BradleyTerryResult[T]:
    """
    Compute the Bradley-Terry scores for the given pairwise comparison.

    Quote:
        Bradley, R.A., Terry, M.E.: Rank Analysis of Incomplete Block Designs: I.
        The Method of Paired Comparisons. Biometrika. 39, 324&ndash;345 (1952).
        <https://doi.org/10.2307/2334029>.

    Quote:
        Newman, M.E.J.: Efficient Computation of Rankings from Pairwise Comparisons.
        Journal of Machine Learning Research. 24, 1&ndash;25 (2023).
        <https://www.jmlr.org/papers/v24/22-1086.html>.

    Args:
        xs: The left-hand side elements.
        ys: The right-hand side elements.
        winners: The winner elements.
        index: The index.
        weights: The example weights.
        win_weight: The win weight.
        tie_weight: The tie weight.
        solver: The solver.
        tolerance: The convergence tolerance.
        limit: The maximum number of iterations.

    Returns:
        The Bradley-Terry result.

    """
    assert np.isfinite(win_weight), "win_weight must be finite"
    assert np.isfinite(tie_weight), "tie_weight must be finite"

    xs_indexed, ys_indexed, index = indexing(xs, ys, index)

    assert index is not None, "index is None"

    weights = _wrap_weights(weights, len(xs_indexed))

    if solver == "pyo3":
        scores, iterations = bradley_terry_pyo3(
            xs=xs_indexed,
            ys=ys_indexed,
            winners=winners,
            weights=weights,
            total=len(index),
            win_weight=win_weight,
            tie_weight=tie_weight,
            tolerance=tolerance,
            limit=limit,
        )
    else:
        _matrices = matrices(
            xs_indexed=xs_indexed,
            ys_indexed=ys_indexed,
            winners=winners,
            index=index,
            weights=weights,
        )

        matrix = _make_matrix(_matrices.win_matrix, _matrices.tie_matrix, win_weight, tie_weight, tolerance)

        scores, iterations = bradley_terry_naive(
            matrix=matrix,
            tolerance=tolerance,
            limit=limit,
        )

    return BradleyTerryResult(
        scores=pd.Series(scores, index=index, name=bradley_terry.__name__).sort_values(ascending=False, kind="stable"),
        index=index,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver=solver,
        tolerance=tolerance,
        iterations=iterations,
        limit=limit,
    )


@dataclass(frozen=True)
class NewmanResult(Generic[T]):
    """
    The Newman's algorithm result.

    Attributes:
        scores: The element scores.
        index: The index.
        v: The tie parameter.
        v_init: The initial tie parameter.
        win_weight: The win weight.
        tie_weight: The tie weight.
        solver: The solver.
        tolerance: The convergence tolerance.
        iterations: The actual number of iterations.
        limit: The maximum number of iterations.

    """

    scores: pd.Series[float]
    index: dict[T, int]
    v: float
    v_init: float
    win_weight: float
    tie_weight: float
    solver: str
    tolerance: float
    iterations: int
    limit: int


def newman(
        xs: Collection[T],
        ys: Collection[T],
        winners: Collection[Winner],
        index: dict[T, int] | None = None,
        v_init: float = .5,
        weights: Collection[float] | None = None,
        win_weight: float = 1.,
        tie_weight: float = 1.,
        solver: Literal["naive", "pyo3"] = "pyo3",
        tolerance: float = 1e-6,
        limit: int = 100,
) -> NewmanResult[T]:
    """
    Compute the scores for the given pairwise comparison using the Newman's algorithm.

    Quote:
        Newman, M.E.J.: Efficient Computation of Rankings from Pairwise Comparisons.
        Journal of Machine Learning Research. 24, 1&ndash;25 (2023).
        <https://www.jmlr.org/papers/v24/22-1086.html>.

    Args:
        xs: The left-hand side elements.
        ys: The right-hand side elements.
        winners: The winner elements.
        index: The index.
        v_init: The initial tie parameter.
        weights: The example weights.
        win_weight: The win weight.
        tie_weight: The tie weight.
        solver: The solver.
        tolerance: The convergence tolerance.
        limit: The maximum number of iterations.

    Returns:
        The Newman's result.

    """
    assert np.isfinite(win_weight), "win_weight must be finite"
    assert np.isfinite(tie_weight), "tie_weight must be finite"

    xs_indexed, ys_indexed, index = indexing(xs, ys, index)

    assert index is not None, "index is None"

    weights = _wrap_weights(weights, len(xs_indexed))

    if solver == "pyo3":
        scores, v, iterations = newman_pyo3(
            xs=xs_indexed,
            ys=ys_indexed,
            winners=winners,
            weights=weights,
            total=len(index),
            v_init=v_init,
            win_weight=win_weight,
            tie_weight=tie_weight,
            tolerance=tolerance,
            limit=limit,
        )

    else:
        _matrices = matrices(
            xs_indexed=xs_indexed,
            ys_indexed=ys_indexed,
            winners=winners,
            index=index,
            weights=weights,
        )

        win_matrix = np.nan_to_num(win_weight * np.nan_to_num(_matrices.win_matrix, nan=tolerance), nan=tolerance)
        tie_matrix = np.nan_to_num(tie_weight * np.nan_to_num(_matrices.tie_matrix, nan=tolerance), nan=tolerance)

        scores, v, iterations = newman_naive(
            win_matrix=win_matrix,
            tie_matrix=tie_matrix,
            v=v_init,
            tolerance=tolerance,
            limit=limit,
        )

    return NewmanResult(
        scores=pd.Series(scores, index=index, name=newman.__name__).sort_values(ascending=False, kind="stable"),
        index=index,
        v=v,
        v_init=v_init,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver=solver,
        tolerance=tolerance,
        iterations=iterations,
        limit=limit,
    )


@dataclass(frozen=True)
class EloResult(Generic[T]):
    """
    The Elo result.

    Attributes:
        scores: The element scores.
        index: The index.
        initial: The initial score of each element.
        base: The base of the exponent.
        scale: The scale factor.
        k: The K-factor.
        win_weight: The win weight.
        tie_weight: The tie weight.
        solver: The solver.

    """

    scores: pd.Series[float]
    index: dict[T, int]
    initial: float
    base: float
    scale: float
    k: float
    win_weight: float
    tie_weight: float
    solver: str


def elo(
        xs: Collection[T],
        ys: Collection[T],
        winners: Collection[Winner],
        index: dict[T, int] | None = None,
        initial: float = 1000.,
        base: float = 10.,
        scale: float = 400.,
        k: float = 4.,
        weights: Collection[float] | None = None,
        win_weight: float = 1.0,
        tie_weight: float = 0.5,
        solver: Literal["naive", "pyo3"] = "pyo3",
) -> EloResult[T]:
    """
    Compute the Elo scores.

    Quote:
        Elo, A.E.: The rating of chessplayers, past and present. Arco Pub, New York (1978).

    Args:
        xs: The left-hand side elements.
        ys: The right-hand side elements.
        winners: The winner elements.
        index: The index.
        initial: The initial score of each element.
        base: The base of the exponent.
        scale: The scale factor.
        k: The K-factor.
        weights: The example weights.
        win_weight: The win weight.
        tie_weight: The tie weight.
        solver: The solver.

    Returns:
        The Elo result.

    """
    xs_indexed, ys_indexed, index = indexing(xs, ys, index)

    assert index is not None, "index is None"

    weights = _wrap_weights(weights, len(xs_indexed))

    if solver == "pyo3":
        scores = elo_pyo3(
            xs=xs_indexed,
            ys=ys_indexed,
            winners=winners,
            weights=weights,
            total=len(index),
            initial=initial,
            base=base,
            scale=scale,
            k=k,
            win_weight=win_weight,
            tie_weight=tie_weight,
        )
    else:
        scores = elo_naive(
            xs=xs_indexed,
            ys=ys_indexed,
            winners=winners,
            weights=weights,
            total=len(index),
            initial=initial,
            base=base,
            scale=scale,
            k=k,
            win_weight=win_weight,
            tie_weight=tie_weight,
        )

    return EloResult(
        scores=pd.Series(scores, index=index, name=elo.__name__).sort_values(ascending=False, kind="stable"),
        index=index,
        initial=initial,
        base=base,
        scale=scale,
        k=k,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver=solver,
    )


@dataclass(frozen=True)
class EigenResult(Generic[T]):
    """
    The eigenvalue result.

    Attributes:
        scores: The element scores.
        index: The index.
        win_weight: The win weight.
        tie_weight: The tie weight.
        solver: The solver.
        tolerance: The convergence tolerance.
        iterations: The actual number of iterations.
        limit: The maximum number of iterations.

    """

    scores: pd.Series[float]
    index: dict[T, int]
    win_weight: float
    tie_weight: float
    solver: str
    tolerance: float
    iterations: int
    limit: int


def eigen(
        xs: Collection[T],
        ys: Collection[T],
        winners: Collection[Winner],
        index: dict[T, int] | None = None,
        weights: Collection[float] | None = None,
        win_weight: float = 1.,
        tie_weight: float = .5,
        solver: Literal["naive", "pyo3"] = "pyo3",
        tolerance: float = 1e-6,
        limit: int = 100,
) -> EigenResult[T]:
    """
    Compute the eigenvalue-based scores.

    Args:
        xs: The left-hand side elements.
        ys: The right-hand side elements.
        winners: The winner elements.
        index: The index.
        weights: The example weights.
        win_weight: The win weight.
        tie_weight: The tie weight.
        solver: The solver.
        tolerance: The convergence tolerance.
        limit: The maximum number of iterations.

    Returns:
        The eigenvalue result.

    """
    assert np.isfinite(win_weight), "win_weight must be finite"
    assert np.isfinite(tie_weight), "tie_weight must be finite"

    xs_indexed, ys_indexed, index = indexing(xs, ys, index)

    assert index is not None, "index is None"

    weights = _wrap_weights(weights, len(xs_indexed))

    if solver == "pyo3":
        scores, iterations = eigen_pyo3(
            xs=xs_indexed,
            ys=ys_indexed,
            winners=winners,
            weights=weights,
            total=len(index),
            win_weight=win_weight,
            tie_weight=tie_weight,
            tolerance=tolerance,
            limit=limit,
        )
    else:
        _matrices = matrices(
            xs_indexed=xs_indexed,
            ys_indexed=ys_indexed,
            winners=winners,
            index=index,
            weights=weights,
        )

        matrix = _make_matrix(_matrices.win_matrix, _matrices.tie_matrix, win_weight, tie_weight, tolerance)

        scores, iterations = eigen_naive(
            matrix=matrix,
            tolerance=tolerance,
            limit=limit,
        )

    return EigenResult(
        scores=pd.Series(scores, index=index, name=eigen.__name__).sort_values(ascending=False, kind="stable"),
        index=index,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver=solver,
        tolerance=tolerance,
        iterations=iterations,
        limit=limit,
    )


@dataclass(frozen=True)
class PageRankResult(Generic[T]):
    """
    The PageRank result.

    Attributes:
        scores: The element scores.
        index: The index.
        damping: The damping (alpha) factor.
        win_weight: The win weight.
        tie_weight: The tie weight.
        solver: The solver.
        tolerance: The convergence tolerance.
        iterations: The actual number of iterations.
        limit: The maximum number of iterations.

    """

    scores: pd.Series[float]
    index: dict[T, int]
    damping: float
    win_weight: float
    tie_weight: float
    solver: str
    tolerance: float
    iterations: int
    limit: int


def pagerank(
        xs: Collection[T],
        ys: Collection[T],
        winners: Collection[Winner],
        index: dict[T, int] | None = None,
        damping: float = .85,
        weights: Collection[float] | None = None,
        win_weight: float = 1.,
        tie_weight: float = .5,
        solver: Literal["naive", "pyo3"] = "pyo3",
        tolerance: float = 1e-6,
        limit: int = 100,
) -> PageRankResult[T]:
    """
    Compute the PageRank scores.

    Quote:
        Brin, S., Page, L.: The anatomy of a large-scale hypertextual Web search engine.
        Computer Networks and ISDN Systems. 30, 107&ndash;117 (1998).
        <https://doi.org/10.1016/S0169-7552(98)00110-X>.

    Args:
        xs: The left-hand side elements.
        ys: The right-hand side elements.
        winners: The winner elements.
        index: The index.
        damping: The damping (alpha) factor.
        weights: The example weights.
        win_weight: The win weight.
        tie_weight: The tie weight.
        solver: The solver.
        tolerance: The convergence tolerance.
        limit: The maximum number of iterations.

    Returns:
        The PageRank result.

    """
    assert np.isfinite(win_weight), "win_weight must be finite"
    assert np.isfinite(tie_weight), "tie_weight must be finite"

    xs_indexed, ys_indexed, index = indexing(xs, ys, index)

    assert index is not None, "index is None"

    weights = _wrap_weights(weights, len(xs_indexed))

    if solver == "pyo3":
        scores, iterations = pagerank_pyo3(
            xs=xs_indexed,
            ys=ys_indexed,
            winners=winners,
            weights=weights,
            total=len(index),
            damping=damping,
            win_weight=win_weight,
            tie_weight=tie_weight,
            tolerance=tolerance,
            limit=limit,
        )
    else:
        _matrices = matrices(
            xs_indexed=xs_indexed,
            ys_indexed=ys_indexed,
            winners=winners,
            index=index,
            weights=weights,
        )

        matrix = _make_matrix(_matrices.win_matrix, _matrices.tie_matrix, win_weight, tie_weight, tolerance)

        scores, iterations = pagerank_naive(
            matrix=matrix,
            damping=damping,
            tolerance=tolerance,
            limit=limit,
        )

    return PageRankResult(
        scores=pd.Series(scores, index=index, name=pagerank.__name__).sort_values(ascending=False, kind="stable"),
        index=index,
        damping=damping,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver=solver,
        tolerance=tolerance,
        iterations=iterations,
        limit=limit,
    )


class ScoreDimensionError(ValueError):
    """Inappropriate dimension given; it should be 1D."""

    def __init__(self, ndim: int) -> None:
        """
        Create and return a new object.

        Args:
            ndim: The given number of dimensions.

        """
        super().__init__(f"scores should be one-dimensional, {ndim} was provided")


def pairwise_scores(
        scores: npt.NDArray[np.float64],
        solver: Literal["naive", "pyo3"] = "pyo3",
) -> npt.NDArray[np.float64]:
    """
    Estimate the pairwise scores.

    Args:
        scores: The element scores.
        solver: The solver.

    Returns:
        The matrix representing pairwise scores between the elements.

    """
    if scores.ndim != 1:
        raise ScoreDimensionError(scores.ndim)

    if solver == "naive":
        return pairwise_scores_naive(scores)

    return pairwise_scores_pyo3(scores)


def pairwise_frame(scores: pd.Series[float]) -> pd.DataFrame:
    """
    Create a data frame out of the estimated pairwise scores.

    Args:
        scores: The element scores.

    Returns:
        The data frame representing pairwise scores between the elements.

    """
    return pd.DataFrame(pairwise_scores(scores.to_numpy()), index=scores.index, columns=scores.index)


__all__ = [
    "BradleyTerryResult",
    "CountingResult",
    "EigenResult",
    "EloResult",
    "LengthMismatchError",
    "MatricesResult",
    "NewmanResult",
    "PageRankResult",
    "Result",
    "ScoreDimensionError",
    "WINNERS",
    "Winner",
    "__version__",
    "average_win_rate",
    "bradley_terry",
    "counting",
    "eigen",
    "elo",
    "indexing",
    "matrices",
    "newman",
    "pagerank",
    "pairwise_frame",
    "pairwise_scores",
]
