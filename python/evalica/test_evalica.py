from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis.extra._array_helpers import array_shapes
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import series
from pandas._testing import assert_series_equal

import evalica
from conftest import Comparison, comparisons

if TYPE_CHECKING:
    import pandas as pd


def test_version() -> None:
    assert isinstance(evalica.__version__, str)
    assert len(evalica.__version__) > 0


def test_exports() -> None:
    for attr in evalica.__all__:
        assert hasattr(evalica, attr), f"missing attribute: {attr}"


def test_winner_pickle() -> None:
    for w in evalica.WINNERS:
        dumped = pickle.dumps(w)
        loaded = pickle.loads(dumped)  # noqa: S301
        assert w == loaded


@given(comparison=comparisons())
def test_indexing(comparison: Comparison) -> None:  # type: ignore[type-var]
    xs, ys, _ = comparison

    xs_indexed, ys_indexed, index = evalica.indexing(xs, ys)

    assert len(xs_indexed) == len(xs)
    assert len(ys_indexed) == len(ys)
    assert isinstance(index, dict)
    assert len(index) == len(set(xs) | set(ys))
    assert set(index.values()) == (set(xs_indexed) | set(ys_indexed))


@given(comparison=comparisons())
def test_reindexing(comparison: Comparison) -> None:
    xs, ys, _ = comparison

    xs_indexed, ys_indexed, index = evalica.indexing(xs, ys)
    xs_reindexed, ys_reindexed, reindex = evalica.indexing(xs, ys, index)

    assert xs_reindexed == xs_indexed
    assert ys_reindexed == ys_indexed
    assert reindex is index


@given(comparison=comparisons())
def test_reindexing_unknown(comparison: Comparison) -> None:
    xs, ys, _ = comparison

    xs_indexed, ys_indexed, index = evalica.indexing(xs, ys)

    xs += [" ".join(xs) + "_unknown"]
    ys += [" ".join(ys) + "_unknown"]

    with pytest.raises(TypeError):
        evalica.indexing(xs, ys, index)


@given(comparison=comparisons())
def test_matrices(comparison: Comparison) -> None:
    xs, ys, ws = comparison

    xs_indexed, ys_indexed, index = evalica.indexing(xs, ys)

    wins = sum(status in [evalica.Winner.X, evalica.Winner.Y] for status in ws)
    ties = sum(status == evalica.Winner.Draw for status in ws)

    result = evalica.matrices(xs_indexed, ys_indexed, ws, index)

    assert result.win_matrix.shape == (len(index), len(index))
    assert result.tie_matrix.shape == (len(index), len(index))
    assert result.win_matrix.sum() == wins
    assert result.tie_matrix.sum() == 2 * ties


@given(comparison=comparisons(), win_weight=st.floats(0., 10.), tie_weight=st.floats(0., 10.))
def test_counting(comparison: Comparison, win_weight: float, tie_weight: float) -> None:
    xs, ys, ws = comparison

    result_pyo3 = evalica.counting(
        xs, ys, ws,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver="pyo3",
    )

    result_naive = evalica.counting(
        xs, ys, ws,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver="naive",
    )

    for result in (result_pyo3, result_naive):
        assert len(result.scores) == len(set(xs) | set(ys))
        assert np.isfinite(result.scores).all()
        assert result.scores.is_monotonic_decreasing

    assert_series_equal(result_pyo3.scores, result_naive.scores, check_like=True)


@given(comparison=comparisons(), win_weight=st.floats(0., 10.), tie_weight=st.floats(0., 10.))
def test_average_win_rate(comparison: Comparison, win_weight: float, tie_weight: float) -> None:
    xs, ys, ws = comparison

    result_pyo3 = evalica.average_win_rate(
        xs, ys, ws,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver="pyo3",
    )

    result_naive = evalica.average_win_rate(
        xs, ys, ws,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver="naive",
    )

    for result in (result_pyo3, result_naive):
        assert len(result.scores) == len(set(xs) | set(ys))
        assert np.isfinite(result.scores).all()
        assert result.scores.is_monotonic_decreasing

    assert_series_equal(result_pyo3.scores, result_naive.scores, check_like=True)


@given(comparison=comparisons(), win_weight=st.floats(0., 10.), tie_weight=st.floats(0., 10.))
def test_bradley_terry(comparison: Comparison, win_weight: float, tie_weight: float) -> None:
    xs, ys, ws = comparison

    result_pyo3 = evalica.bradley_terry(
        xs, ys, ws,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver="pyo3",
    )

    result_naive = evalica.bradley_terry(
        xs, ys, ws,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver="naive",
    )

    for result in (result_pyo3, result_naive):
        assert len(result.scores) == len(set(xs) | set(ys))
        assert np.isfinite(result.scores).all()
        assert result.scores.is_monotonic_decreasing
        assert result.iterations > 0
        assert result.limit > 0

    assert_series_equal(result_pyo3.scores, result_naive.scores, rtol=1e-4, check_like=True)


@given(comparison=comparisons(), v_init=st.floats())
def test_newman(comparison: Comparison, v_init: float) -> None:
    xs, ys, ws = comparison

    result_pyo3 = evalica.newman(xs, ys, ws, v_init=v_init, solver="pyo3")
    result_naive = evalica.newman(xs, ys, ws, v_init=v_init, solver="naive")

    for result in (result_pyo3, result_naive):
        assert len(result.scores) == len(set(xs) | set(ys))
        assert np.isfinite(result.scores).all()
        assert result.scores.is_monotonic_decreasing
        assert np.isfinite(result.v)
        assert result.iterations > 0
        assert result.limit > 0

        if np.isfinite(v_init):
            assert result.v_init == v_init
        else:
            assert result.v_init is v_init

    assert_series_equal(result_pyo3.scores, result_naive.scores, check_like=True)
    assert result_pyo3.v == pytest.approx(result_naive.v)


@given(
    comparison=comparisons(),
    initial=st.floats(0., 1000.),
    base=st.floats(0., 1000.),
    scale=st.floats(0., 1000.),
    k=st.floats(0., 1000.),
)
def test_elo(
        comparison: Comparison,
        initial: float,
        base: float,
        scale: float,
        k: float,
) -> None:
    xs, ys, ws = comparison

    result_pyo3 = evalica.elo(
        xs, ys, ws,
        initial=initial,
        base=base,
        scale=scale,
        k=k,
        solver="pyo3",
    )

    result_naive = evalica.elo(
        xs, ys, ws,
        initial=initial,
        base=base,
        scale=scale,
        k=k,
        solver="naive",
    )

    for result in (result_pyo3, result_naive):
        assert len(result.scores) == len(set(xs) | set(ys))
        assert np.isfinite(result.scores).all()
        assert result.scores.is_monotonic_decreasing

    assert_series_equal(result_pyo3.scores, result_naive.scores, check_like=True)


@given(comparison=comparisons(), win_weight=st.floats(0., 10.), tie_weight=st.floats(0., 10.))
def test_eigen(comparison: Comparison, win_weight: float, tie_weight: float) -> None:
    xs, ys, ws = comparison

    result_pyo3 = evalica.eigen(
        xs, ys, ws,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver="pyo3",
    )

    result_naive = evalica.eigen(
        xs, ys, ws,
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

    assert_series_equal(result_pyo3.scores, result_naive.scores, check_like=True)


@given(
    comparison=comparisons(),
    damping=st.floats(0., 1.),
    win_weight=st.floats(0., 10.),
    tie_weight=st.floats(0., 10.),
)
def test_pagerank(comparison: Comparison, damping: float, win_weight: float, tie_weight: float) -> None:
    xs, ys, ws = comparison

    result_pyo3 = evalica.pagerank(
        xs, ys, ws,
        damping=damping,
        win_weight=win_weight,
        tie_weight=tie_weight,
        solver="pyo3",
    )

    result_naive = evalica.pagerank(
        xs, ys, ws,
        damping=damping,
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

    assert_series_equal(result_pyo3.scores, result_naive.scores, check_like=True)


@given(comparison=comparisons(shape="bad"))
@pytest.mark.parametrize(("algorithm", "solver"), [
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
])
def test_misshaped(comparison: Comparison, algorithm: str, solver: str) -> None:
    with pytest.raises(evalica.LengthMismatchError):
        getattr(evalica, algorithm)(*comparison, solver=solver)


@pytest.mark.parametrize(("algorithm", "solver"), [
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
])
def test_incomplete_index(algorithm: str, solver: str) -> None:
    xs = ["a", "c", "e"]
    ys = ["b", "d", "f"]
    ws = [evalica.Winner.X, evalica.Winner.Ignore, evalica.Winner.Y]

    _, _, index = evalica.indexing(xs, ys)

    result = getattr(evalica, algorithm)(xs, ys, ws, index=index, solver=solver)
    result_incomplete = getattr(evalica, algorithm)(xs[:-1], ys[:-1], ws[:-1], index=index, solver=solver)

    assert len(result.scores) == len(result_incomplete.scores)


@pytest.mark.parametrize(("algorithm", "dataset"), [
    ("counting", "simple"),
    ("counting", "food"),
    ("counting", "llmfao"),
])
def test_counting_dataset(comparison: Comparison, comparison_golden: pd.Series[str]) -> None:
    xs, ys, ws = comparison

    result_pyo3 = evalica.counting(xs, ys, ws, solver="pyo3")
    result_naive = evalica.counting(xs, ys, ws, solver="naive")

    assert_series_equal(result_naive.scores, comparison_golden, check_like=True)
    assert_series_equal(result_pyo3.scores, comparison_golden, check_like=True)
    assert_series_equal(result_pyo3.scores, result_naive.scores, check_like=True)


@pytest.mark.parametrize(("algorithm", "dataset"), [
    ("average_win_rate", "simple"),
    ("average_win_rate", "food"),
    ("average_win_rate", "llmfao"),
])
def test_average_win_rate_dataset(comparison: Comparison, comparison_golden: pd.Series[str]) -> None:
    xs, ys, ws = comparison

    result_pyo3 = evalica.average_win_rate(xs, ys, ws, solver="pyo3")
    result_naive = evalica.average_win_rate(xs, ys, ws, solver="naive")

    assert_series_equal(result_naive.scores, comparison_golden, check_like=True)
    assert_series_equal(result_pyo3.scores, comparison_golden, check_like=True)
    assert_series_equal(result_pyo3.scores, result_naive.scores, check_like=True)


@pytest.mark.parametrize(("algorithm", "dataset"), [
    ("bradley_terry", "simple"),
    ("bradley_terry", "food"),
    ("bradley_terry", "llmfao"),
])
def test_bradley_terry_dataset(comparison: Comparison, comparison_golden: pd.Series[str]) -> None:
    xs, ys, ws = comparison

    result_pyo3 = evalica.bradley_terry(xs, ys, ws, solver="pyo3")
    result_naive = evalica.bradley_terry(xs, ys, ws, solver="naive")

    assert_series_equal(result_naive.scores, comparison_golden, rtol=1e-4, check_like=True)
    assert_series_equal(result_pyo3.scores, comparison_golden, rtol=1e-4, check_like=True)
    assert_series_equal(result_pyo3.scores, result_naive.scores, check_like=True)


@pytest.mark.parametrize(("algorithm", "dataset"), [
    ("newman", "simple"),
    ("newman", "food"),
    ("newman", "llmfao"),
])
def test_newman_dataset(comparison: Comparison, comparison_golden: pd.Series[str]) -> None:
    xs, ys, ws = comparison

    result_pyo3 = evalica.newman(xs, ys, ws, solver="pyo3")
    result_naive = evalica.newman(xs, ys, ws, solver="naive")

    assert_series_equal(result_naive.scores, comparison_golden, rtol=1e-4, check_like=True)
    assert_series_equal(result_pyo3.scores, comparison_golden, rtol=1e-4, check_like=True)

    assert_series_equal(result_pyo3.scores, result_naive.scores, check_like=True)
    assert result_pyo3.v == pytest.approx(result_naive.v)


@pytest.mark.parametrize(("algorithm", "dataset"), [
    ("elo", "simple"),
    ("elo", "food"),
    ("elo", "llmfao"),
])
def test_elo_dataset(comparison: Comparison, comparison_golden: pd.Series[str]) -> None:
    xs, ys, ws = comparison

    result_pyo3 = evalica.elo(xs, ys, ws, solver="pyo3")
    result_naive = evalica.elo(xs, ys, ws, solver="naive")

    assert_series_equal(result_naive.scores, comparison_golden, check_like=True)
    assert_series_equal(result_pyo3.scores, comparison_golden, check_like=True)
    assert_series_equal(result_pyo3.scores, result_naive.scores, check_like=True)


@pytest.mark.parametrize(("algorithm", "dataset"), [
    ("eigen", "simple"),
    ("eigen", "food"),
    ("eigen", "llmfao"),
])
def test_eigen_dataset(comparison: Comparison, comparison_golden: pd.Series[str]) -> None:
    xs, ys, ws = comparison

    result_pyo3 = evalica.eigen(xs, ys, ws, solver="pyo3")
    result_naive = evalica.eigen(xs, ys, ws, solver="naive")

    assert_series_equal(result_naive.scores, comparison_golden, check_like=True)
    assert_series_equal(result_pyo3.scores, comparison_golden, check_like=True)
    assert_series_equal(result_pyo3.scores, result_naive.scores, check_like=True)


@pytest.mark.parametrize(("algorithm", "dataset"), [
    ("pagerank", "simple"),
    ("pagerank", "food"),
    ("pagerank", "llmfao"),
])
def test_pagerank_dataset(comparison: Comparison, comparison_golden: pd.Series[str]) -> None:
    xs, ys, ws = comparison

    result_pyo3 = evalica.pagerank(xs, ys, ws, solver="pyo3")
    result_naive = evalica.pagerank(xs, ys, ws, solver="naive")

    assert_series_equal(result_naive.scores, comparison_golden, check_like=True)
    assert_series_equal(result_pyo3.scores, comparison_golden, check_like=True)
    assert_series_equal(result_pyo3.scores, result_naive.scores, check_like=True)


@given(arrays(dtype=np.float64, shape=array_shapes(max_dims=1, min_side=0)))
def test_pairwise_scores(scores: npt.NDArray[np.float64]) -> None:
    with np.errstate(all="ignore"):
        pairwise = evalica.pairwise_scores(scores)

    assert pairwise.dtype == scores.dtype
    assert pairwise.shape == (len(scores), len(scores))

    if np.isfinite(scores).all():
        assert np.isfinite(pairwise).all()
    else:
        assert not np.isfinite(pairwise).all()


def test_pairwise_scores_empty() -> None:
    pairwise = evalica.pairwise_scores(np.zeros(0, dtype=np.float64))
    assert pairwise.dtype == np.float64
    assert pairwise.shape == (0, 0)


@given(array_shapes())
def test_pairwise_scores_shape(shape: tuple[int, ...]) -> None:
    scores = np.zeros(shape, dtype=np.int64)

    if len(shape) == 1:
        with np.errstate(all="ignore"):
            evalica.pairwise_scores(scores)
    else:
        with pytest.raises(evalica.ScoreDimensionError):
            evalica.pairwise_scores(scores)


@given(series(dtype=np.float64))
def test_pairwise_frame(scores: pd.Series[Any]) -> None:
    with np.errstate(all="ignore"):
        df_pairwise = evalica.pairwise_frame(scores)

    assert df_pairwise.shape == (len(scores), len(scores))
    assert df_pairwise.index is scores.index
    assert df_pairwise.columns is scores.index
