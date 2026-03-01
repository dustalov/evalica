from __future__ import annotations

import warnings
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from hypothesis import assume, given
from hypothesis.extra._array_helpers import array_shapes
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import series
from numpy.testing import assert_array_equal
from pandas._testing import assert_series_equal

import evalica
from conftest import Comparison, comparisons
from evalica import SolverName

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture


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

    wins = sum(int(winner in [evalica.Winner.X, evalica.Winner.Y]) * weight for winner, weight in zip(winners, weights))
    ties = sum(int(winner == evalica.Winner.Draw) * weight for winner, weight in zip(winners, weights))

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
def test_pairwise_frame(scores: pd.Series[Any]) -> None:
    with np.errstate(all="ignore"):
        df_pairwise = evalica.pairwise_frame(scores)

    assert df_pairwise.shape == (len(scores), len(scores))
    assert df_pairwise.index is scores.index
    assert df_pairwise.columns is scores.index


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

    assert_array_equal(result.result.scores.index.sort_values(), result.index.sort_values())
    assert_array_equal(result.low.index, result.index)
    assert_array_equal(result.high.index, result.index)
    assert_array_equal(result.stderr.index, result.index)
    assert_array_equal(result.distribution.columns, result.index)

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
            cast("evalica.RankingMethod[str]", evalica.counting),
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
            cast("evalica.RankingMethod[str]", evalica.counting),
            comparison.xs,
            comparison.ys,
            comparison.winners,
            weights=weights,
            n_resamples=n_resamples,
            random_state=42,
        )

    assert len(result.distribution) == n_resamples
