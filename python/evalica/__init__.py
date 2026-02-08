"""Evalica, your favourite evaluation suite."""

from __future__ import annotations

import contextlib
import math
import os
import warnings
from collections.abc import Hashable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import bootstrap as scipy_bootstrap


class Winner(IntEnum):
    """The outcome of the pairwise comparison."""

    Draw = 0
    """There is a tie."""

    X = 1
    """The first element won."""

    Y = 2
    """The second element won."""


class LengthMismatchError(ValueError):
    """The dataset dimensions mismatched."""


class SolverError(RuntimeError):
    """The requested solver is not available."""

    def __init__(self, solver: str) -> None:
        """
        Create and return a new object.

        Args:
            solver: The solver name.

        """
        super().__init__(f"The '{solver}' solver is not available")


class InsufficientRatingsError(ValueError):
    """Raised when no units have at least 2 ratings."""

    def __init__(self) -> None:
        """Create and return a new object."""
        super().__init__("No units have at least 2 ratings.")


class UnknownDistanceError(ValueError):
    """Raised when an unknown distance metric is specified."""

    def __init__(self, distance: str) -> None:
        """
        Create and return a new object.

        Args:
            distance: The distance metric name.

        """
        super().__init__(f"Unknown distance '{distance}'")


class RustExtensionWarning(RuntimeWarning):
    """The Rust extension could not be imported."""


try:
    if os.environ.get("EVALICA_NIJE_BRZO"):
        raise ImportError  # noqa: TRY301

    from . import _brzo
    from ._brzo import __version__

    PYO3_AVAILABLE = True
    """
    The Rust extension is available and can be used for performance-critical operations.

    Please set the environment variable EVALICA_NIJE_BRZO to disable it.
    """
except ImportError:
    warnings.warn(
        "The Rust extension could not be imported; falling back to the naive implementations.",
        RustExtensionWarning,
        stacklevel=1,
    )

    import importlib.metadata

    with contextlib.suppress(importlib.metadata.PackageNotFoundError):
        __version__ = importlib.metadata.version("evalica")

    _brzo = None  # type: ignore[assignment]

    PYO3_AVAILABLE = False


@dataclass(frozen=True)
class AlphaResult:
    """
    The result of Krippendorff's alpha.

    Attributes:
        alpha: The alpha value.
        observed: The observed disagreement.
        expected: The expected disagreement.
        solver: The solver used.

    """

    alpha: float
    observed: float
    expected: float
    solver: str


T_distance_contra = TypeVar("T_distance_contra", contravariant=True)


DistanceName = Literal["interval", "nominal", "ordinal", "ratio"]


class DistanceFunc(Protocol[T_distance_contra]):
    """
    Callable protocol for custom distance functions.

    Args:
        left: The left-hand value.
        right: The right-hand value.

    Returns:
        The non-negative distance between the values.

    """

    def __call__(self, left: T_distance_contra, right: T_distance_contra, /) -> float: ...


SOLVER: Literal["naive", "pyo3"] = "pyo3" if PYO3_AVAILABLE else "naive"
"""The default solver."""

from .alpha import _alpha_naive  # noqa: E402
from .pairwise import bradley_terry as bradley_terry_naive  # noqa: E402
from .pairwise import counting as counting_naive  # noqa: E402
from .pairwise import eigen as eigen_naive  # noqa: E402
from .pairwise import elo as elo_naive  # noqa: E402
from .pairwise import matrices as matrices_naive  # noqa: E402
from .pairwise import newman as newman_naive  # noqa: E402
from .pairwise import pagerank as pagerank_naive  # noqa: E402
from .pairwise import pairwise_scores as pairwise_scores_naive  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Collection

WINNERS = list(Winner)
"""Known values of Winner."""

T_contra = TypeVar("T_contra", bound=Hashable, contravariant=True)


def _wrap_weights(weights: Collection[float] | npt.NDArray[np.float64] | None, n: int) -> Collection[float]:
    if weights is None:
        return [1.0] * n

    if isinstance(weights, np.ndarray):
        weights = weights.tolist()

    assert isinstance(weights, list), "weights must be a list"

    assert all(math.isfinite(w) for w in weights), "weights must be finite"

    return weights


def _make_matrix(
    win_matrix: npt.NDArray[np.float64],
    tie_matrix: npt.NDArray[np.float64],
    win_weight: float = 1.0,
    tie_weight: float = 0.5,
    nan: float = 0.0,
) -> npt.NDArray[np.float64]:
    with np.errstate(all="ignore"):
        return np.nan_to_num(
            win_weight * np.nan_to_num(win_matrix, nan=nan) + tie_weight * np.nan_to_num(tie_matrix, nan=nan),
            nan=nan,
        )


def indexing(
    xs: Collection[T_contra],
    ys: Collection[T_contra],
    index: pd.Index | None = None,
) -> tuple[list[int], list[int], pd.Index]:
    """
    Map the input elements into their numerical representations.

    Args:
        xs: The left-hand side elements.
        ys: The right-hand side elements.
        index: The index; if provided, all elements in xs and ys must be present in it.

    Returns:
        The tuple containing the numerical representations of the input elements and the corresponding index.

    """
    if index is None:
        labels = list(dict.fromkeys([*xs, *ys]))
        index = pd.Index(labels)

    xi = index.get_indexer(cast("pd.Index", xs))
    yi = index.get_indexer(cast("pd.Index", ys))

    if (xi < 0).any() or (yi < 0).any():
        msg = "Unknown element in reindexing"
        raise TypeError(msg)

    return xi.tolist(), yi.tolist(), index


@dataclass(frozen=True)
class MatricesResult:
    """
    The win and tie matrices.

    Attributes:
        win_matrix: The matrix representing wins between the elements.
        tie_matrix: The matrix representing ties between the elements; it is always symmetric.
        index: The index.

    """

    win_matrix: npt.NDArray[np.float64]
    tie_matrix: npt.NDArray[np.float64]
    index: pd.Index


def matrices(
    xs_indexed: Collection[int],
    ys_indexed: Collection[int],
    winners: Collection[Winner],
    index: pd.Index,
    weights: Collection[float] | None = None,
    solver: Literal["naive", "pyo3"] = SOLVER,
) -> MatricesResult:
    """
    Build win and tie matrices from the given elements.

    Args:
        xs_indexed: The left-hand side elements.
        ys_indexed: The right-hand side elements.
        winners: The winner elements.
        index: The index.
        weights: The example weights.
        solver: The solver.

    Returns:
        The win and tie matrices.

    """
    weights = _wrap_weights(weights, len(xs_indexed))

    if solver == "pyo3":
        if not PYO3_AVAILABLE:
            raise SolverError(solver)

        win_matrix, tie_matrix = _brzo.matrices(
            xs=xs_indexed,
            ys=ys_indexed,
            winners=winners,
            weights=weights,
            total=len(index),
        )
    else:
        win_matrix, tie_matrix = matrices_naive(
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
class Result(Protocol):
    """
    The result protocol.

    Attributes:
        scores: The element scores.
        index: The index.

    """

    scores: pd.Series[float]
    index: pd.Index


@runtime_checkable
class RankingMethod(Protocol[T_contra]):
    """The ranking method protocol."""

    def __call__(
        self,
        xs: Collection[T_contra],
        ys: Collection[T_contra],
        winners: Collection[Winner],
        index: pd.Index | None = None,
        weights: Collection[float] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Result:
        """
        Compute the scores for the given pairwise comparison.

        Args:
            xs: The left-hand side elements.
            ys: The right-hand side elements.
            winners: The winner elements.
            index: The index.
            weights: The example weights.
            *args: The additional positional arguments.
            **kwargs: The additional keyword arguments.

        Returns:
            The ranking result.

        """
        ...


@dataclass(frozen=True)
class CountingResult:
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
    index: pd.Index
    win_weight: float
    tie_weight: float
    solver: str


def counting(
    xs: Collection[T_contra],
    ys: Collection[T_contra],
    winners: Collection[Winner],
    index: pd.Index | None = None,
    weights: Collection[float] | None = None,
    win_weight: float = 1.0,
    tie_weight: float = 0.5,
    solver: Literal["naive", "pyo3"] = SOLVER,
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> CountingResult:
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
        **kwargs: The additional arguments.

    Returns:
        The counting result.

    """
    assert np.isfinite(win_weight), "win_weight must be finite"
    assert np.isfinite(tie_weight), "tie_weight must be finite"

    xs_indexed, ys_indexed, index = indexing(xs, ys, index)

    assert index is not None, "index is None"

    weights = _wrap_weights(weights, len(xs_indexed))

    if solver == "pyo3":
        if not PYO3_AVAILABLE:
            raise SolverError(solver)

        scores = _brzo.counting(
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
class AverageWinRateResult:
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
    index: pd.Index
    win_weight: float
    tie_weight: float
    solver: str


def average_win_rate(
    xs: Collection[T_contra],
    ys: Collection[T_contra],
    winners: Collection[Winner],
    index: pd.Index | None = None,
    weights: Collection[float] | None = None,
    win_weight: float = 1.0,
    tie_weight: float = 0.5,
    solver: Literal["naive", "pyo3"] = SOLVER,
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> AverageWinRateResult:
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
        **kwargs: The additional arguments.

    Returns:
        The average win rate result.

    """
    assert np.isfinite(win_weight), "win_weight must be finite"
    assert np.isfinite(tie_weight), "tie_weight must be finite"

    xs_indexed, ys_indexed, index = indexing(xs, ys, index)

    assert index is not None, "index is None"

    weights = _wrap_weights(weights, len(xs_indexed))

    if solver == "pyo3":
        if not PYO3_AVAILABLE:
            raise SolverError(solver)

        scores = _brzo.average_win_rate(
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
            solver="naive",
        )

        matrix = _make_matrix(_matrices.win_matrix, _matrices.tie_matrix, win_weight, tie_weight)

        with np.errstate(all="ignore"):
            denominator = np.nan_to_num(matrix + matrix.T)

            matrix /= denominator

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Mean of empty slice")

            scores = np.nan_to_num(np.nanmean(matrix, axis=1), copy=False)

    return AverageWinRateResult(
        scores=pd.Series(
            scores,
            index=index,
            name=average_win_rate.__name__,
        ).sort_values(ascending=False, kind="stable"),
        index=index,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver=solver,
    )


@dataclass(frozen=True)
class BradleyTerryResult:
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
    index: pd.Index
    win_weight: float
    tie_weight: float
    solver: str
    tolerance: float
    iterations: int
    limit: int


def bradley_terry(
    xs: Collection[T_contra],
    ys: Collection[T_contra],
    winners: Collection[Winner],
    index: pd.Index | None = None,
    weights: Collection[float] | None = None,
    win_weight: float = 1.0,
    tie_weight: float = 0.5,
    solver: Literal["naive", "pyo3"] = SOLVER,
    tolerance: float = 1e-6,
    limit: int = 100,
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> BradleyTerryResult:
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
        **kwargs: The additional arguments.

    Returns:
        The Bradley-Terry result.

    """
    assert np.isfinite(win_weight), "win_weight must be finite"
    assert np.isfinite(tie_weight), "tie_weight must be finite"

    xs_indexed, ys_indexed, index = indexing(xs, ys, index)

    assert index is not None, "index is None"

    weights = _wrap_weights(weights, len(xs_indexed))

    if solver == "pyo3":
        if not PYO3_AVAILABLE:
            raise SolverError(solver)

        scores, iterations = _brzo.bradley_terry(
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
            solver="naive",
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
class NewmanResult:
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
    index: pd.Index
    v: float
    v_init: float
    win_weight: float
    tie_weight: float
    solver: str
    tolerance: float
    iterations: int
    limit: int


def newman(
    xs: Collection[T_contra],
    ys: Collection[T_contra],
    winners: Collection[Winner],
    index: pd.Index | None = None,
    v_init: float = 0.5,
    weights: Collection[float] | None = None,
    win_weight: float = 1.0,
    tie_weight: float = 1.0,
    solver: Literal["naive", "pyo3"] = SOLVER,
    tolerance: float = 1e-6,
    limit: int = 100,
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> NewmanResult:
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
        **kwargs: The additional arguments.

    Returns:
        The Newman's result.

    """
    assert np.isfinite(win_weight), "win_weight must be finite"
    assert np.isfinite(tie_weight), "tie_weight must be finite"

    xs_indexed, ys_indexed, index = indexing(xs, ys, index)

    assert index is not None, "index is None"

    weights = _wrap_weights(weights, len(xs_indexed))

    if solver == "pyo3":
        if not PYO3_AVAILABLE:
            raise SolverError(solver)

        scores, v, iterations = _brzo.newman(
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
            solver="naive",
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
class EloResult:
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
    index: pd.Index
    initial: float
    base: float
    scale: float
    k: float
    win_weight: float
    tie_weight: float
    solver: str


def elo(
    xs: Collection[T_contra],
    ys: Collection[T_contra],
    winners: Collection[Winner],
    index: pd.Index | None = None,
    initial: float = 1000.0,
    base: float = 10.0,
    scale: float = 400.0,
    k: float = 4.0,
    weights: Collection[float] | None = None,
    win_weight: float = 1.0,
    tie_weight: float = 0.5,
    solver: Literal["naive", "pyo3"] = SOLVER,
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> EloResult:
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
        **kwargs: The additional arguments.

    Returns:
        The Elo result.

    """
    xs_indexed, ys_indexed, index = indexing(xs, ys, index)

    assert index is not None, "index is None"

    weights = _wrap_weights(weights, len(xs_indexed))

    if solver == "pyo3":
        if not PYO3_AVAILABLE:
            raise SolverError(solver)

        scores = _brzo.elo(
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
class EigenResult:
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
    index: pd.Index
    win_weight: float
    tie_weight: float
    solver: str
    tolerance: float
    iterations: int
    limit: int


def eigen(
    xs: Collection[T_contra],
    ys: Collection[T_contra],
    winners: Collection[Winner],
    index: pd.Index | None = None,
    weights: Collection[float] | None = None,
    win_weight: float = 1.0,
    tie_weight: float = 0.5,
    solver: Literal["naive", "pyo3"] = SOLVER,
    tolerance: float = 1e-6,
    limit: int = 100,
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> EigenResult:
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
        **kwargs: The additional arguments.

    Returns:
        The eigenvalue result.

    """
    assert np.isfinite(win_weight), "win_weight must be finite"
    assert np.isfinite(tie_weight), "tie_weight must be finite"

    xs_indexed, ys_indexed, index = indexing(xs, ys, index)

    assert index is not None, "index is None"

    weights = _wrap_weights(weights, len(xs_indexed))

    if solver == "pyo3":
        if not PYO3_AVAILABLE:
            raise SolverError(solver)

        scores, iterations = _brzo.eigen(
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
            solver="naive",
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
class PageRankResult:
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
    index: pd.Index
    damping: float
    win_weight: float
    tie_weight: float
    solver: str
    tolerance: float
    iterations: int
    limit: int


def pagerank(
    xs: Collection[T_contra],
    ys: Collection[T_contra],
    winners: Collection[Winner],
    index: pd.Index | None = None,
    damping: float = 0.85,
    weights: Collection[float] | None = None,
    win_weight: float = 1.0,
    tie_weight: float = 0.5,
    solver: Literal["naive", "pyo3"] = SOLVER,
    tolerance: float = 1e-6,
    limit: int = 100,
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> PageRankResult:
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
        **kwargs: The additional arguments.

    Returns:
        The PageRank result.

    """
    assert np.isfinite(win_weight), "win_weight must be finite"
    assert np.isfinite(tie_weight), "tie_weight must be finite"

    xs_indexed, ys_indexed, index = indexing(xs, ys, index)

    assert index is not None, "index is None"

    weights = _wrap_weights(weights, len(xs_indexed))

    if solver == "pyo3":
        if not PYO3_AVAILABLE:
            raise SolverError(solver)

        scores, iterations = _brzo.pagerank(
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
            solver="naive",
        )

        matrix = _make_matrix(_matrices.win_matrix, _matrices.tie_matrix, win_weight, tie_weight, tolerance)

        scores, iterations = pagerank_naive(
            matrix=matrix,
            damping=damping,
            tolerance=tolerance,
            limit=limit,
        )

    return PageRankResult(
        scores=pd.Series(data=scores, index=index, name=pagerank.__name__).sort_values(ascending=False, kind="stable"),
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
    solver: Literal["naive", "pyo3"] = SOLVER,
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

    if solver == "pyo3":
        if not PYO3_AVAILABLE:
            raise SolverError(solver)

        return _brzo.pairwise_scores(scores)

    return pairwise_scores_naive(scores)


def pairwise_frame(scores: pd.Series[float]) -> pd.DataFrame:
    """
    Create a data frame out of the estimated pairwise scores.

    Args:
        scores: The element scores.

    Returns:
        The data frame representing pairwise scores between the elements.

    """
    arr = np.asarray(scores.array, dtype=np.float64)
    return pd.DataFrame(pairwise_scores(arr), index=scores.index, columns=scores.index)


@dataclass(frozen=True)
class BootstrapResult:
    """
    The result of a bootstrap operation.

    Attributes:
        result: The original point estimates (from the full dataset).
        low: Lower bounds of the confidence interval.
        high: Upper bounds of the confidence interval.
        stderr: Standard errors of the scores.
        distribution: The full bootstrap distribution (resamples x elements).
        index: The index of elements.

    """

    result: Result
    low: pd.Series[float]
    high: pd.Series[float]
    stderr: pd.Series[float]
    distribution: pd.DataFrame = field(repr=False)
    index: pd.Index


def bootstrap(
    method: RankingMethod[T_contra],
    xs: Collection[T_contra],
    ys: Collection[T_contra],
    winners: Collection[Winner],
    weights: Collection[float] | None = None,
    index: pd.Index | None = None,
    win_weight: float = 1.0,
    tie_weight: float = 0.5,
    solver: Literal["naive", "pyo3"] = SOLVER,
    *,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    bootstrap_method: Literal["percentile", "basic", "BCa"] = "BCa",
    random_state: int | np.random.Generator | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> BootstrapResult:
    """
    Compute weighted bootstrap confidence intervals for the given pairwise comparison.

    Args:
        xs: The left-hand side elements.
        ys: The right-hand side elements.
        winners: The winner elements.
        weights: The example weights.
        method: The ranking method to use.
        index: The index.
        win_weight: The win weight.
        tie_weight: The tie weight.
        solver: The solver.
        n_resamples: The number of resamples.
        confidence_level: The confidence level.
        bootstrap_method: The bootstrap method (percentile, basic, or BCa).
        random_state: The random state.
        **kwargs: The additional arguments for the ranking method.

    Returns:
        The bootstrap result.

    """
    _, _, index = indexing(xs, ys, index)

    result = method(
        xs=xs,
        ys=ys,
        winners=winners,
        weights=weights,
        index=index,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver=solver,
        **kwargs,
    )

    weights_array = np.array(_wrap_weights(weights, len(xs)), dtype=np.float64)

    samples = (np.array(xs, dtype=object), np.array(ys, dtype=object), np.array(winners, dtype=np.uint8), weights_array)

    def statistic(*data: tuple[Any, ...]) -> npt.NDArray[np.float64]:
        xs_sample, ys_sample, winners_sample, weights_sample = data

        result_sample = method(
            xs=xs_sample,
            ys=ys_sample,
            winners=list(winners_sample),  # TODO: ensure no copying needed
            index=index,
            weights=weights_sample,
            win_weight=win_weight,
            tie_weight=tie_weight,
            solver=solver,
            **kwargs,
        )

        return cast("npt.NDArray[np.float64]", result_sample.scores.to_numpy(dtype=np.float64))

    bootstrap_result = scipy_bootstrap(
        data=samples,
        statistic=statistic,
        paired=True,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=bootstrap_method,
        random_state=random_state,
        vectorized=False,
    )

    return BootstrapResult(
        result=result,
        low=pd.Series(bootstrap_result.confidence_interval.low, index=index, name="low"),
        high=pd.Series(bootstrap_result.confidence_interval.high, index=index, name="high"),
        stderr=pd.Series(bootstrap_result.standard_error, index=index, name="stderr"),
        distribution=pd.DataFrame(bootstrap_result.bootstrap_distribution.T, columns=index),
        index=index,
    )


def alpha(
    data: pd.DataFrame,
    distance: DistanceFunc[T_distance_contra] | DistanceName = "nominal",
    solver: Literal["naive", "pyo3"] = "pyo3",
) -> AlphaResult:
    """
    Compute Krippendorff's alpha.

    Args:
        data: Ratings by observer (rows) and unit (columns).
        distance: Distance metric (nominal, ordinal, interval, ratio) or a custom function.
        solver: The solver to use (naive or pyo3).

    Returns:
        The alpha result.

    Raises:
        InsufficientRatingsError: If no units have at least 2 ratings.
        UnknownDistanceError: If an unknown distance metric is specified.
        SolverError: If the requested solver is not available or incompatible.

    """
    if solver == "pyo3" and (not PYO3_AVAILABLE or callable(distance)):
        raise SolverError(solver)

    if solver == "pyo3":
        data_array = data.T.to_numpy(dtype=float)

        assert not callable(distance), "distance must not be a function"

        _alpha, observed, expected = _brzo.alpha(data_array, distance)
    else:
        _alpha, observed, expected = _alpha_naive(data, distance)

    return AlphaResult(
        alpha=_alpha,
        observed=observed,
        expected=expected,
        solver=solver,
    )


__all__ = [
    "PYO3_AVAILABLE",
    "SOLVER",
    "WINNERS",
    "AverageWinRateResult",
    "BootstrapResult",
    "BradleyTerryResult",
    "CountingResult",
    "EigenResult",
    "EloResult",
    "InsufficientRatingsError",
    "LengthMismatchError",
    "MatricesResult",
    "NewmanResult",
    "PageRankResult",
    "RankingMethod",
    "Result",
    "RustExtensionWarning",
    "ScoreDimensionError",
    "SolverError",
    "UnknownDistanceError",
    "Winner",
    "__version__",
    "alpha",
    "average_win_rate",
    "bootstrap",
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
