from __future__ import annotations

import sys
import warnings
from collections.abc import Collection
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from hypothesis import assume, example, given
from hypothesis.extra._array_helpers import array_shapes
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import series
from numpy.testing import assert_array_equal
from pandas._testing import assert_series_equal

import evalica
from conftest import Comparison, comparisons
from evalica import SolverName

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_codspeed import BenchmarkFixture


@given(labels=st.lists(st.text(min_size=1), min_size=2, max_size=10, unique=True))
def test_bootstrap_score_order(labels: list[str]) -> None:
    xs = labels
    ys = labels[1:] + labels[:1]
    winners = [evalica.Winner.X] * len(labels)

    # We use reversed order to ensure it's often non-alphabetical
    expected_order = labels[::-1]

    def mock_method(**kwargs: object) -> evalica.CountingResult:
        index = cast("pd.Index", kwargs["index"])
        scores = pd.Series(
            np.arange(len(labels), 0, -1, dtype=float),
            index=expected_order,
            name="mock",
        )
        return evalica.CountingResult(scores=scores, index=index, win_weight=1.0, tie_weight=0.5, solver="naive")

    result = evalica.bootstrap(
        cast("evalica.RankingMethod[str]", mock_method),
        xs,
        ys,
        winners,
        n_resamples=5,
        bootstrap_method="percentile",
        random_state=42,
    )

    assert list(result.low.index) == expected_order
    assert list(result.high.index) == expected_order
    assert list(result.stderr.index) == expected_order
    assert list(result.distribution.columns) == expected_order


@given(comparison=comparisons())
def test_indexing(comparison: Comparison) -> None:
    xs, ys, *_ = comparison

    xs_indexed, ys_indexed, index = evalica.indexing(xs, ys)

    assert len(xs_indexed) == len(xs)
    assert len(ys_indexed) == len(ys)
    assert isinstance(index, pd.Index)
    assert len(index) == len(set(xs) | set(ys))
    assert set(index.tolist()) == (set(xs) | set(ys))
    assert set(xs_indexed) | set(ys_indexed) == set(index.get_indexer(index))


@given(comparison=comparisons())
def test_reindexing(comparison: Comparison) -> None:
    xs, ys, *_ = comparison

    xs_indexed, ys_indexed, index = evalica.indexing(xs, ys)
    xs_reindexed, ys_reindexed, reindex = evalica.indexing(xs, ys, index)

    assert xs_reindexed == xs_indexed
    assert ys_reindexed == ys_indexed
    assert reindex is index


@given(comparison=comparisons())
def test_reindexing_unknown(comparison: Comparison) -> None:
    xs, ys, *_ = comparison

    *_, index = evalica.indexing(xs, ys)

    xs += [" ".join(xs) + "_unknown"]
    ys += [" ".join(ys) + "_unknown"]

    with pytest.raises(TypeError):
        evalica.indexing(xs, ys, index)


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
@given(comparison=comparisons())
def test_matrices(comparison: Comparison, solver: SolverName) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    xs, ys, winners, weights = comparison

    xs_indexed, ys_indexed, index = evalica.indexing(xs, ys)

    if weights is None:
        weights = [1.0] * len(winners)

    wins = sum(
        int(winner in [evalica.Winner.X, evalica.Winner.Y]) * weight
        for winner, weight in zip(winners, weights, strict=True)
    )
    ties = sum(int(winner == evalica.Winner.Draw) * weight for winner, weight in zip(winners, weights, strict=True))

    assume(np.isfinite(wins) and np.isfinite(ties))

    result = evalica.matrices(
        xs_indexed=xs_indexed,
        ys_indexed=ys_indexed,
        winners=winners,
        index=index,
        weights=weights,
        solver=solver,
    )

    assert result.win_matrix.shape == (len(index), len(index))
    assert result.tie_matrix.shape == (len(index), len(index))

    with np.errstate(over="ignore"):
        assert result.win_matrix.sum() == pytest.approx(wins)
        assert result.tie_matrix.sum() == pytest.approx(2 * ties)


@pytest.mark.skipif(not evalica.PYO3_AVAILABLE, reason="Rust extension is not available")
@given(comparison=comparisons(), win_weight=st.floats(0.0, 10.0), tie_weight=st.floats(0.0, 10.0))
def test_counting(comparison: Comparison, win_weight: float, tie_weight: float) -> None:
    xs, ys, winners, weights = comparison

    result_pyo3 = evalica.counting(
        xs,
        ys,
        winners,
        weights=weights,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver="pyo3",
    )

    result_naive = evalica.counting(
        xs,
        ys,
        winners,
        weights=weights,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver="naive",
    )

    for result in (result_pyo3, result_naive):
        assert len(result.scores) == len(set(xs) | set(ys))
        assert np.isfinite(result.scores).all()
        assert result.scores.is_monotonic_decreasing
        assert isinstance(result, evalica.Result)

    assert_series_equal(result_pyo3.scores, result_naive.scores, check_like=True)


@pytest.mark.skipif(not evalica.PYO3_AVAILABLE, reason="Rust extension is not available")
@given(comparison=comparisons(), win_weight=st.floats(0.0, 10.0), tie_weight=st.floats(0.0, 10.0))
def test_average_win_rate(comparison: Comparison, win_weight: float, tie_weight: float) -> None:
    xs, ys, winners, weights = comparison

    result_pyo3 = evalica.average_win_rate(
        xs,
        ys,
        winners,
        weights=weights,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver="pyo3",
    )

    result_naive = evalica.average_win_rate(
        xs,
        ys,
        winners,
        weights=weights,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver="naive",
    )

    for result in (result_pyo3, result_naive):
        assert len(result.scores) == len(set(xs) | set(ys))
        assert np.isfinite(result.scores).all()
        assert result.scores.is_monotonic_decreasing
        assert isinstance(result, evalica.Result)

    assert_series_equal(result_pyo3.scores, result_naive.scores, rtol=1e-4, check_like=True)


@pytest.mark.skipif(not evalica.PYO3_AVAILABLE, reason="Rust extension is not available")
@given(comparison=comparisons(), win_weight=st.floats(0.0, 10.0), tie_weight=st.floats(0.0, 10.0))
def test_bradley_terry(comparison: Comparison, win_weight: float, tie_weight: float) -> None:
    xs, ys, winners, weights = comparison

    result_pyo3 = evalica.bradley_terry(
        xs,
        ys,
        winners,
        weights=weights,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver="pyo3",
    )

    result_naive = evalica.bradley_terry(
        xs,
        ys,
        winners,
        weights=weights,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver="naive",
    )

    for result in (result_pyo3, result_naive):
        assert len(result.scores) == len(set(xs) | set(ys))
        assert np.isfinite(result.scores).all()
        assert result.scores.is_monotonic_decreasing
        assert not xs or result.iterations > 0
        assert result.limit > 0
        assert isinstance(result, evalica.Result)

    assert_series_equal(result_pyo3.scores, result_naive.scores, check_like=True)


@pytest.mark.skipif(not evalica.PYO3_AVAILABLE, reason="Rust extension is not available")
@given(comparison=comparisons(), v_init=st.floats())
def test_newman(comparison: Comparison, v_init: float) -> None:
    xs, ys, winners, weights = comparison

    result_pyo3 = evalica.newman(
        xs,
        ys,
        winners,
        v_init=v_init,
        weights=weights,
        solver="pyo3",
    )

    result_naive = evalica.newman(
        xs,
        ys,
        winners,
        v_init=v_init,
        weights=weights,
        solver="naive",
    )

    for result in (result_pyo3, result_naive):
        assert len(result.scores) == len(set(xs) | set(ys))
        assert np.isfinite(result.scores).all()
        assert result.scores.is_monotonic_decreasing
        assert not xs or np.isfinite(result.v)
        assert not xs or result.iterations > 0
        assert result.v_init is v_init
        assert result.limit > 0

        assert isinstance(result, evalica.Result)

    assert_series_equal(result_pyo3.scores, result_naive.scores, check_like=True)

    assert not np.isfinite(v_init) or result_pyo3.v == pytest.approx(result_naive.v)


@pytest.mark.skipif(not evalica.PYO3_AVAILABLE, reason="Rust extension is not available")
@given(
    comparison=comparisons(),
    initial=st.floats(0.0, 1000.0),
    base=st.floats(0.0, 1000.0),
    scale=st.floats(0.0, 1000.0),
    k=st.floats(0.0, 1000.0),
    win_weight=st.floats(0.0, 10.0),
    tie_weight=st.floats(0.0, 10.0),
)
def test_elo(
    comparison: Comparison,
    win_weight: float,
    tie_weight: float,
    initial: float,
    base: float,
    scale: float,
    k: float,
) -> None:
    xs, ys, winners, weights = comparison

    result_pyo3 = evalica.elo(
        xs,
        ys,
        winners,
        initial=initial,
        base=base,
        scale=scale,
        k=k,
        weights=weights,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver="pyo3",
    )

    result_naive = evalica.elo(
        xs,
        ys,
        winners,
        initial=initial,
        base=base,
        scale=scale,
        k=k,
        weights=weights,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver="naive",
    )

    for result in (result_pyo3, result_naive):
        assert len(result.scores) == len(set(xs) | set(ys))
        assert np.isfinite(result.scores).all()
        assert result.scores.is_monotonic_decreasing
        assert isinstance(result, evalica.Result)

    assert_series_equal(result_pyo3.scores, result_naive.scores, check_like=True)


@pytest.mark.skipif(not evalica.PYO3_AVAILABLE, reason="Rust extension is not available")
@given(comparison=comparisons(), win_weight=st.floats(0.0, 10.0), tie_weight=st.floats(0.0, 10.0))
def test_eigen(comparison: Comparison, win_weight: float, tie_weight: float) -> None:
    xs, ys, winners, weights = comparison

    result_pyo3 = evalica.eigen(
        xs,
        ys,
        winners,
        weights=weights,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver="pyo3",
    )

    result_naive = evalica.eigen(
        xs,
        ys,
        winners,
        weights=weights,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver="naive",
    )

    for result in (result_pyo3, result_naive):
        assert len(result.scores) == len(set(xs) | set(ys))
        assert np.isfinite(result.scores).all()
        assert result.scores.is_monotonic_decreasing
        assert not xs or result.iterations > 0
        assert result.limit > 0
        assert isinstance(result, evalica.Result)

    assert_series_equal(result_pyo3.scores, result_naive.scores, check_like=True)


@pytest.mark.skipif(not evalica.PYO3_AVAILABLE, reason="Rust extension is not available")
@given(
    comparison=comparisons(),
    damping=st.floats(0.0, 1.0),
    win_weight=st.floats(0.0, 10.0),
    tie_weight=st.floats(0.0, 10.0),
)
def test_pagerank(comparison: Comparison, damping: float, win_weight: float, tie_weight: float) -> None:
    xs, ys, winners, weights = comparison

    result_pyo3 = evalica.pagerank(
        xs,
        ys,
        winners,
        damping=damping,
        weights=weights,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver="pyo3",
    )

    result_naive = evalica.pagerank(
        xs,
        ys,
        winners,
        damping=damping,
        weights=weights,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver="naive",
    )

    for result in (result_pyo3, result_naive):
        assert len(result.scores) == len(set(xs) | set(ys))
        assert np.isfinite(result.scores).all()
        assert result.scores.is_monotonic_decreasing
        assert np.isfinite(result.damping)
        assert np.isfinite(result.win_weight)
        assert np.isfinite(result.tie_weight)
        assert not xs or result.iterations > 0
        assert result.limit > 0
        assert isinstance(result, evalica.Result)

    assert_series_equal(result_pyo3.scores, result_naive.scores, check_like=True)


@given(comparison=comparisons(shape="bad"))
@pytest.mark.parametrize(
    ("algorithm", "solver"),
    [
        ("counting", "pyo3"),
        ("counting", "naive"),
        ("average_win_rate", "pyo3"),
        ("average_win_rate", "naive"),
        ("bradley_terry", "pyo3"),
        ("bradley_terry", "naive"),
        ("newman", "pyo3"),
        ("newman", "naive"),
        ("elo", "pyo3"),
        ("elo", "naive"),
        ("eigen", "pyo3"),
        ("eigen", "naive"),
        ("pagerank", "pyo3"),
        ("pagerank", "naive"),
    ],
)
def test_misshaped(comparison: Comparison, algorithm: str, solver: SolverName) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    with pytest.raises(evalica.LengthMismatchError):
        getattr(evalica, algorithm)(comparison.xs, comparison.ys, comparison.winners, solver=solver)


@pytest.mark.parametrize(
    ("algorithm", "solver"),
    [
        ("counting", "naive"),
        ("counting", "pyo3"),
        ("average_win_rate", "naive"),
        ("average_win_rate", "pyo3"),
        ("bradley_terry", "naive"),
        ("bradley_terry", "pyo3"),
        ("newman", "naive"),
        ("newman", "pyo3"),
        ("elo", "naive"),
        ("elo", "pyo3"),
        ("eigen", "naive"),
        ("eigen", "pyo3"),
        ("pagerank", "naive"),
        ("pagerank", "pyo3"),
    ],
)
def test_incomplete_index(algorithm: str, solver: SolverName) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    xs = ["a", "c", "e"]
    ys = ["b", "d", "f"]
    winners = [evalica.Winner.X, evalica.Winner.Draw, evalica.Winner.Y]

    _, _, index = evalica.indexing(xs, ys)

    result = getattr(evalica, algorithm)(xs, ys, winners, index=index, solver=solver)
    result_incomplete = getattr(evalica, algorithm)(xs[:-1], ys[:-1], winners[:-1], index=index, solver=solver)

    assert len(result.scores) == len(result_incomplete.scores)


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
@pytest.mark.parametrize(
    ("algorithm", "dataset"),
    [
        ("counting", "simple"),
        ("counting", "food"),
        ("counting", "llmfao"),
    ],
)
def test_counting_dataset(
    comparison: Comparison,
    comparison_golden: pd.Series[str],
    solver: SolverName,
) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    xs, ys, winners, weights = comparison

    result = evalica.counting(xs, ys, winners, weights=weights, solver=solver)

    assert_series_equal(result.scores, comparison_golden, check_like=True)


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
@pytest.mark.parametrize(
    ("algorithm", "dataset"),
    [
        ("average_win_rate", "simple"),
        ("average_win_rate", "food"),
        ("average_win_rate", "llmfao"),
    ],
)
def test_average_win_rate_dataset(
    comparison: Comparison,
    comparison_golden: pd.Series[str],
    solver: SolverName,
) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    xs, ys, winners, weights = comparison

    result = evalica.average_win_rate(xs, ys, winners, weights=weights, solver=solver)

    assert_series_equal(result.scores, comparison_golden, check_like=True)


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
@pytest.mark.parametrize(
    ("algorithm", "dataset"),
    [
        ("bradley_terry", "simple"),
        ("bradley_terry", "food"),
        ("bradley_terry", "llmfao"),
    ],
)
def test_bradley_terry_dataset(
    comparison: Comparison,
    comparison_golden: pd.Series[str],
    solver: SolverName,
) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    xs, ys, winners, weights = comparison

    result = evalica.bradley_terry(xs, ys, winners, weights=weights, solver=solver)

    scores = result.scores / result.scores.sum()

    assert_series_equal(scores, comparison_golden, rtol=1e-4, check_like=True)


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
@pytest.mark.parametrize(
    ("algorithm", "dataset"),
    [
        ("newman", "simple"),
        ("newman", "food"),
        ("newman", "llmfao"),
    ],
)
def test_newman_dataset(
    comparison: Comparison,
    comparison_golden: pd.Series[str],
    solver: SolverName,
) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    xs, ys, winners, weights = comparison

    result = evalica.newman(xs, ys, winners, weights=weights, solver=solver)

    assert_series_equal(result.scores, comparison_golden, rtol=1e-4, check_like=True)


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
@pytest.mark.parametrize(
    ("algorithm", "dataset"),
    [
        ("elo", "simple"),
        ("elo", "food"),
        ("elo", "llmfao"),
    ],
)
def test_elo_dataset(
    comparison: Comparison,
    comparison_golden: pd.Series[str],
    solver: SolverName,
) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    xs, ys, winners, weights = comparison

    result = evalica.elo(xs, ys, winners, weights=weights, solver=solver)

    assert_series_equal(result.scores, comparison_golden, check_like=True)


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
@pytest.mark.parametrize(
    ("algorithm", "dataset"),
    [
        ("eigen", "simple"),
        ("eigen", "food"),
        ("eigen", "llmfao"),
    ],
)
def test_eigen_dataset(
    comparison: Comparison,
    comparison_golden: pd.Series[str],
    solver: SolverName,
) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    xs, ys, winners, weights = comparison

    result = evalica.eigen(xs, ys, winners, weights=weights, solver=solver)

    assert_series_equal(result.scores, comparison_golden, check_like=True)


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
@pytest.mark.parametrize(
    ("algorithm", "dataset"),
    [
        ("pagerank", "simple"),
        ("pagerank", "food"),
        ("pagerank", "llmfao"),
    ],
)
def test_pagerank_dataset(
    comparison: Comparison,
    comparison_golden: pd.Series[str],
    solver: SolverName,
) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    xs, ys, winners, weights = comparison

    result = evalica.pagerank(xs, ys, winners, weights=weights, solver=solver)

    assert_series_equal(result.scores, comparison_golden, check_like=True)


@pytest.mark.benchmark
def test_llmfao_indexing(llmfao: Comparison) -> None:
    evalica.indexing(llmfao.xs, llmfao.ys)


def test_llmfao_matrices(llmfao: Comparison, benchmark: BenchmarkFixture) -> None:
    xs_indexed, ys_indexed, index = evalica.indexing(llmfao.xs, llmfao.ys)

    func = partial(evalica.matrices, xs_indexed, ys_indexed, llmfao.winners, weights=llmfao.weights, index=index)

    benchmark(func)


@pytest.mark.parametrize(
    ("algorithm", "solver"),
    [
        ("counting", "naive"),
        ("counting", "pyo3"),
        ("average_win_rate", "naive"),
        ("average_win_rate", "pyo3"),
        ("bradley_terry", "naive"),
        ("bradley_terry", "pyo3"),
        ("newman", "naive"),
        ("newman", "pyo3"),
        ("elo", "naive"),
        ("elo", "pyo3"),
        ("eigen", "naive"),
        ("eigen", "pyo3"),
        ("pagerank", "naive"),
        ("pagerank", "pyo3"),
    ],
)
def test_llmfao_performance(
    llmfao: Comparison,
    algorithm: str,
    solver: SolverName,
    benchmark: BenchmarkFixture,
) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    _, _, index = evalica.indexing(llmfao.xs, llmfao.ys)

    func = partial(
        getattr(evalica, algorithm),
        llmfao.xs,
        llmfao.ys,
        llmfao.winners,
        index=index,
        weights=llmfao.weights,
        solver=solver,
    )

    benchmark(func)


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
def test_llmfao_pairwise_scores(
    llmfao: Comparison,
    solver: Literal["pyo3", "naive"],
    benchmark: BenchmarkFixture,
) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    result = evalica.counting(llmfao.xs, llmfao.ys, llmfao.winners, weights=llmfao.weights)

    func = partial(evalica.pairwise_scores, np.asarray(result.scores.array, dtype=np.float64), solver=solver)

    benchmark(func)


@given(scores=arrays(dtype=np.float64, shape=array_shapes(max_dims=1, min_side=0)))
def test_pairwise_scores_naive(scores: npt.NDArray[np.float64]) -> None:
    with np.errstate(all="ignore"):
        result = evalica.pairwise_scores(scores, solver="naive")

    assert result.dtype == scores.dtype
    assert result.shape == (len(scores), len(scores))
    assert np.isfinite(result).all()


@pytest.mark.skipif(not evalica.PYO3_AVAILABLE, reason="Rust extension is not available")
@given(scores=arrays(dtype=np.float64, shape=array_shapes(max_dims=1, min_side=0)))
def test_pairwise_scores_pyo3(scores: npt.NDArray[np.float64]) -> None:
    with np.errstate(all="ignore"):
        result_pyo3 = evalica.pairwise_scores(scores, solver="pyo3")
        result_naive = evalica.pairwise_scores(scores, solver="naive")

    assert result_pyo3.dtype == scores.dtype
    assert result_pyo3.shape == (len(scores), len(scores))
    assert np.isfinite(result_pyo3).all()
    assert_array_equal(result_pyo3, result_naive)


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
def test_pairwise_scores_empty(solver: Literal["pyo3", "naive"]) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    pairwise = evalica.pairwise_scores(np.zeros(0, dtype=np.float64), solver=solver)

    assert pairwise.dtype == np.float64
    assert pairwise.shape == (0, 0)


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
@given(array_shapes())
def test_pairwise_scores_shape(solver: Literal["pyo3", "naive"], shape: tuple[int, ...]) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    scores = np.zeros(shape)

    if len(shape) == 1:
        with np.errstate(all="ignore"):
            evalica.pairwise_scores(scores, solver=solver)
    else:
        with pytest.raises(ValueError):  # noqa: PT011
            evalica.pairwise_scores(scores, solver=solver)


@given(series(dtype=np.float64))
def test_pairwise_frame(scores: pd.Series[float]) -> None:
    with np.errstate(all="ignore"):
        df_pairwise = evalica.pairwise_frame(scores)

    assert df_pairwise.shape == (len(scores), len(scores))
    assert df_pairwise.index.equals(scores.index)
    assert df_pairwise.columns.equals(scores.index)


@given(comparison=comparisons())
@pytest.mark.parametrize("method", [evalica.counting, evalica.bradley_terry])
def test_bootstrap_error(comparison: Comparison, method: evalica.RankingMethod[str]) -> None:
    error_comparison = Comparison(
        xs=comparison.xs[:1],
        ys=comparison.ys[:1],
        winners=comparison.winners[:1],
    )

    with pytest.raises(ValueError, match=r"each sample in `data` must contain two or more observations along `axis`."):
        evalica.bootstrap(
            method,
            error_comparison.xs,
            error_comparison.ys,
            error_comparison.winners,
            n_resamples=5,
            random_state=42,
        )


@given(comparison=comparisons())
@pytest.mark.parametrize("method", [evalica.counting, evalica.bradley_terry])
def test_bootstrap(comparison: Comparison, method: evalica.RankingMethod[str]) -> None:
    assume(len(comparison.xs) >= 2)

    n_resamples = 5

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")

        result = evalica.bootstrap(
            method,
            comparison.xs,
            comparison.ys,
            comparison.winners,
            n_resamples=n_resamples,
            random_state=42,
        )

    assert_array_equal(result.index, result.result.index)
    assert_array_equal(result.low.index, result.result.scores.index)
    assert_array_equal(result.high.index, result.result.scores.index)
    assert_array_equal(result.stderr.index, result.result.scores.index)
    assert_array_equal(result.distribution.columns, result.result.scores.index)

    assert len(result.distribution) == n_resamples


@given(comparison=comparisons())
def test_bootstrap_weights_error(comparison: Comparison) -> None:
    error_comparison = Comparison(
        xs=comparison.xs[:1],
        ys=comparison.ys[:1],
        winners=comparison.winners[:1],
        weights=comparison.weights[:1] if comparison.weights is not None else None,
    )

    with pytest.raises(ValueError, match=r"each sample in `data` must contain two or more observations along `axis`."):
        evalica.bootstrap(
            evalica.counting,
            error_comparison.xs,
            error_comparison.ys,
            error_comparison.winners,
            weights=[1.0] * len(error_comparison.xs),
            n_resamples=5,
            random_state=42,
        )


@given(comparison=comparisons())
def test_bootstrap_weights(comparison: Comparison) -> None:
    assume(len(comparison.xs) >= 2)

    n_resamples = 5

    weights = [1.0] * len(comparison.xs)

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")

        result = evalica.bootstrap(
            evalica.counting,
            comparison.xs,
            comparison.ys,
            comparison.winners,
            weights=weights,
            n_resamples=n_resamples,
            random_state=42,
        )

    assert len(result.distribution) == n_resamples


@pytest.mark.parametrize("wrap", [list, tuple, np.array, pd.Series])
def test_weights_collections(wrap: type) -> None:
    result = evalica.counting(["a", "b"], ["b", "c"], [evalica.Winner.X] * 2, weights=wrap([1.0, 2.0]))

    assert result.scores.to_dict() == {"a": 1.0, "b": 2.0, "c": 0.0}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"weights": [1.0, float("nan")]}, "weights must be finite"),
        ({"win_weight": float("inf")}, "win_weight must be finite"),
        ({"tie_weight": float("nan")}, "tie_weight must be finite"),
    ],
)
def test_non_finite_weights(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        evalica.counting(["a", "b"], ["b", "c"], [evalica.Winner.X] * 2, **kwargs)


def _dict_values(winners: Collection[evalica.Winner]) -> Collection[evalica.Winner]:
    return dict(enumerate(winners)).values()


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
@pytest.mark.parametrize(
    "wrap",
    [tuple, np.array, partial(np.array, dtype=np.int64), partial(np.array, dtype=np.float64), pd.Series, _dict_values],
)
@given(comparison=comparisons())
def test_winners_collections(
    solver: SolverName,
    wrap: Callable[[Collection[evalica.Winner]], Collection[evalica.Winner]],
    comparison: Comparison,
) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    xs, ys, winners, weights = comparison

    expected = evalica.counting(xs, ys, winners, weights=weights, solver=solver)
    actual = evalica.counting(xs, ys, wrap(winners), weights=weights, solver=solver)

    assert_series_equal(actual.scores, expected.scores)


VALID_WINNERS = tuple(evalica.Winner)

invalid_winner = st.one_of(
    st.integers().filter(lambda value: value not in VALID_WINNERS),
    st.floats().filter(lambda value: value not in VALID_WINNERS),
    st.text(),
    st.none(),
)


@st.composite
def invalid_winners(draw: st.DrawFn) -> tuple[list[str], list[str], Collection[object]]:
    winners: Collection[object]

    if draw(st.booleans()):
        valid = draw(st.lists(st.sampled_from(VALID_WINNERS), max_size=5))
        position = draw(st.integers(0, len(valid)))
        winners = [*valid[:position], draw(invalid_winner), *valid[position:]]
        length = len(winners)
    elif draw(st.booleans()):
        winners = draw(
            arrays(np.int64, st.integers(1, 5)).filter(lambda array: not np.isin(array, VALID_WINNERS).all()),
        )
        length = len(winners)
    else:
        array = draw(
            arrays(
                np.uint8,
                array_shapes(min_dims=0, min_side=1, max_side=3).filter(lambda shape: len(shape) != 1),
                elements=st.sampled_from(VALID_WINNERS),
            ),
        )
        winners = array
        length = array.shape[0] if array.ndim else 1

    xs = draw(st.lists(st.text(), min_size=length, max_size=length))
    ys = draw(st.lists(st.text(), min_size=length, max_size=length))

    return xs, ys, winners


def _matrices(
    xs: Collection[str],
    ys: Collection[str],
    winners: Collection[evalica.Winner],
    solver: SolverName,
) -> evalica.MatricesResult:
    xs_indexed, ys_indexed, index = evalica.indexing(xs, ys)
    return evalica.matrices(xs_indexed, ys_indexed, winners, index=index, solver=solver)


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
@pytest.mark.parametrize(
    "method",
    [
        _matrices,
        evalica.counting,
        evalica.average_win_rate,
        evalica.bradley_terry,
        evalica.newman,
        evalica.elo,
        evalica.eigen,
        evalica.pagerank,
    ],
)
@given(call=invalid_winners())
def test_invalid_winners(
    solver: SolverName,
    method: Callable[..., object],
    call: tuple[list[str], list[str], Collection[object]],
) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    xs, ys, winners = call

    with pytest.raises(evalica.UnknownWinnerError, match="invalid winner value"):
        method(xs, ys, cast("Collection[evalica.Winner]", winners), solver=solver)


EMPTY_EXPECTED: dict[str, tuple[Callable[..., evalica.Result], Callable[[int, dict[str, float]], float]]] = {
    "counting": (evalica.counting, lambda _n, _kwargs: 0.0),
    "average_win_rate": (evalica.average_win_rate, lambda _n, _kwargs: 0.0),
    "bradley_terry": (evalica.bradley_terry, lambda _n, kwargs: kwargs["tolerance"]),
    "newman": (evalica.newman, lambda _n, kwargs: kwargs["tolerance"]),
    "elo": (evalica.elo, lambda _n, kwargs: kwargs["initial"]),
    "eigen": (evalica.eigen, lambda _n, _kwargs: 0.0),
    "pagerank": (evalica.pagerank, lambda n, _kwargs: 1 / n),
}


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
@pytest.mark.parametrize(("method", "expected"), EMPTY_EXPECTED.values(), ids=EMPTY_EXPECTED.keys())
@given(
    labels=st.lists(st.text(), min_size=1, max_size=10, unique=True),
    tolerance=st.floats(1e-12, 1e-2),
    initial=st.floats(allow_nan=False, allow_infinity=False),
)
def test_empty_comparisons_with_index(
    solver: SolverName,
    method: Callable[..., evalica.Result],
    expected: Callable[[int, dict[str, float]], float],
    labels: list[str],
    tolerance: float,
    initial: float,
) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    kwargs = {"tolerance": tolerance, "initial": initial}
    index = pd.Index(labels)

    result = method([], [], [], index=index, solver=solver, **kwargs)

    assert result.scores.index.sort_values().equals(index.sort_values())
    assert result.scores.to_numpy() == pytest.approx([expected(len(labels), kwargs)] * len(labels))


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
@given(labels=st.lists(st.text(), max_size=10, unique=True))
def test_empty_matrices_with_index(solver: SolverName, labels: list[str]) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    result = evalica.matrices([], [], [], index=pd.Index(labels), solver=solver)

    assert result.win_matrix.shape == result.tie_matrix.shape == (len(labels), len(labels))
    assert (result.win_matrix == 0).all()
    assert (result.tie_matrix == 0).all()


PARITY_METHODS: dict[str, Callable[..., evalica.Result]] = {
    "counting": evalica.counting,
    "average_win_rate": evalica.average_win_rate,
    "bradley_terry": evalica.bradley_terry,
    "newman": evalica.newman,
    "elo": evalica.elo,
    "eigen": evalica.eigen,
    "pagerank": evalica.pagerank,
}


def _run_solver(
    method: Callable[..., evalica.Result],
    solver: SolverName,
    *args: object,
    **kwargs: object,
) -> evalica.Result | type[BaseException]:
    try:
        return method(*args, solver=solver, **kwargs)
    except Exception as e:  # noqa: BLE001
        return type(e)


def _assert_solver_parity(method: Callable[..., evalica.Result], *args: object, **kwargs: object) -> None:
    naive = _run_solver(method, "naive", *args, **kwargs)
    pyo3 = _run_solver(method, "pyo3", *args, **kwargs)

    if isinstance(naive, type) or isinstance(pyo3, type):
        assert naive == pyo3
        return

    assert naive.index.equals(pyo3.index)

    np.testing.assert_allclose(
        naive.scores.reindex(naive.index).to_numpy(),
        pyo3.scores.reindex(pyo3.index).to_numpy(),
        rtol=1e-6,
        atol=1e-9,
        equal_nan=True,
    )

    for attr in ("iterations", "v"):
        if hasattr(naive, attr):
            np.testing.assert_allclose(getattr(naive, attr), getattr(pyo3, attr), rtol=1e-6, atol=1e-9, equal_nan=True)


ParityCall = tuple[tuple[Collection[object], Collection[object], Collection[object]], dict[str, object]]


PARITY_WRAPPERS: list[Callable[[Collection[object]], Collection[object]]] = [
    list,
    np.asarray,
    cast("Callable[[Collection[object]], Collection[object]]", pd.Series),
]


@st.composite
def parity_calls(draw: st.DrawFn) -> ParityCall:
    xs, ys, winners, weights = draw(st.one_of(comparisons(), comparisons(shape="bad")))

    labels = list(dict.fromkeys([*xs, *ys]))

    kwargs: dict[str, object] = {
        "weights": draw(
            st.one_of(
                st.just(weights),
                st.lists(st.floats(), min_size=len(winners), max_size=len(winners)),
                st.lists(st.floats(), max_size=6),
            ),
        ),
        "win_weight": draw(st.floats()),
        "tie_weight": draw(st.floats()),
    }

    if draw(st.booleans()):
        known = draw(st.lists(st.sampled_from(labels), unique=True)) if labels else []
        extra = draw(st.lists(st.text().filter(lambda label: label not in labels), max_size=3, unique=True))
        kwargs["index"] = pd.Index([*known, *extra])

    wrap = draw(st.sampled_from(PARITY_WRAPPERS))

    return (wrap(xs), wrap(ys), wrap(winners)), kwargs


parity_params = st.fixed_dictionaries(
    {},
    optional={
        "tolerance": st.floats(),
        "limit": st.one_of(st.integers(-5, 300), st.integers(min_value=sys.maxsize + 1), st.floats()),
        "v_init": st.floats(),
        "damping": st.floats(),
        "initial": st.floats(),
        "base": st.floats(),
        "scale": st.floats(),
        "k": st.floats(),
    },
)


@pytest.mark.skipif(not evalica.PYO3_AVAILABLE, reason="Rust extension is not available")
@pytest.mark.parametrize("method", PARITY_METHODS.values(), ids=PARITY_METHODS.keys())
@given(call=parity_calls(), params=parity_params)
@example(
    call=(
        (["", "0", "1", "00", "000"], ["0", "1", "000", "", "00"], [evalica.Winner.Draw, *[evalica.Winner.Y] * 4]),
        {"win_weight": 2.5},
    ),
    params={"limit": 1000},
)
@example(
    call=(
        (["", ""], ["", "0"], [evalica.Winner.Draw, evalica.Winner.X]),
        {"weights": [5.525978176869229e298, 7.190772539449264e307], "win_weight": 2.5},
    ),
    params={},
)
def test_solver_parity(method: Callable[..., evalica.Result], call: ParityCall, params: dict[str, object]) -> None:
    args, kwargs = call

    _assert_solver_parity(method, *args, **kwargs, **params)


@pytest.mark.parametrize("method", [evalica.bradley_terry, evalica.newman, evalica.eigen, evalica.pagerank])
@given(
    comparison=comparisons(),
    limit=st.one_of(st.integers(max_value=-1), st.integers(min_value=sys.maxsize + 1), st.floats()),
)
def test_invalid_limit(method: Callable[..., evalica.Result], comparison: Comparison, limit: float) -> None:
    xs, ys, winners, weights = comparison

    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        method(xs, ys, winners, weights=weights, limit=limit)


@given(comparison=comparisons(), damping=st.floats().filter(lambda damping: not 0.0 <= damping <= 1.0))
def test_invalid_damping(comparison: Comparison, damping: float) -> None:
    xs, ys, winners, weights = comparison

    with pytest.raises(ValueError, match="damping must be between 0 and 1"):
        evalica.pagerank(xs, ys, winners, weights=weights, damping=damping)


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
@pytest.mark.parametrize("name", ["weights", "win_weight", "tie_weight"])
@given(
    comparison=comparisons().filter(lambda comparison: len(comparison.xs) > 0),
    negative=st.floats(max_value=0.0, exclude_max=True, allow_infinity=False),
    data=st.data(),
)
def test_pagerank_negative_weights(
    solver: SolverName,
    name: str,
    comparison: Comparison,
    negative: float,
    data: st.DataObject,
) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    xs, ys, winners, weights = comparison

    kwargs: dict[str, Any] = {"weights": weights}

    if name == "weights":
        weights = list(weights) if weights is not None else [1.0] * len(xs)
        weights[data.draw(st.integers(0, len(weights) - 1))] = negative
        kwargs["weights"] = weights
    else:
        kwargs[name] = negative

    with pytest.raises(ValueError, match=f"{name} must be non-negative"):
        evalica.pagerank(xs, ys, winners, solver=solver, **kwargs)


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
@pytest.mark.parametrize("method", [evalica.eigen, evalica.pagerank])
def test_empty_iterations(method: Callable[..., evalica.Result], solver: SolverName) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    result = method([], [], [], solver=solver)
    assert result.iterations == 0  # type: ignore[attr-defined]
