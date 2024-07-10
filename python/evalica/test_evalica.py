import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from pandas._testing import assert_series_equal

import evalica
from conftest import Example, elements


def test_version() -> None:
    assert isinstance(evalica.__version__, str)
    assert len(evalica.__version__) > 0


def test_exports() -> None:
    for attr in evalica.__all__:
        assert hasattr(evalica, attr), f"missing attribute: {attr}"


@given(example=elements())
def test_index_elements(example: Example) -> None:  # type: ignore[type-var]
    xs, ys, ws = example

    indexed = evalica.index_elements(xs, ys)

    assert len(indexed.xs) == len(xs)
    assert len(indexed.ys) == len(ys)
    assert isinstance(indexed.index, pd.Index)
    assert len(indexed.index) == len(set(xs) | set(ys))
    assert set(indexed.index.values) == (set(xs) | set(ys))


@given(example=elements())
def test_matrices(example: Example) -> None:
    xs, ys, ws = example

    n = len(set(xs) | set(ys))

    wins = sum(status in [evalica.Winner.X, evalica.Winner.Y] for status in ws)
    ties = sum(status == evalica.Winner.Draw for status in ws)

    result = evalica.matrices(xs, ys, ws)

    assert result.win_matrix.shape == (n, n)
    assert result.tie_matrix.shape == (n, n)
    assert result.win_matrix.sum() == wins
    assert result.tie_matrix.sum() == 2 * ties


@given(example=elements())
def test_counting(example: Example) -> None:
    xs, ys, ws = example

    result = evalica.counting(xs, ys, ws)

    assert len(result.scores) == len(set(xs) | set(ys))
    assert np.isfinite(result.scores).all()


@given(example=elements())
def test_bradley_terry(example: Example) -> None:
    xs, ys, ws = example

    result_pyo3 = evalica.bradley_terry(xs, ys, ws, solver="pyo3")
    result_naive = evalica.bradley_terry(xs, ys, ws, solver="naive")

    for result in (result_pyo3, result_naive):
        assert result.matrix.shape[0] == len(result.scores)
        assert np.isfinite(result.scores).all()
        assert result.iterations > 0

    tolerance = result_pyo3.tolerance * 10

    assert_series_equal(result_pyo3.scores, result_naive.scores, atol=tolerance)


@given(example=elements())
def test_newman(example: Example) -> None:
    xs, ys, ws = example

    result_pyo3 = evalica.newman(xs, ys, ws, solver="pyo3")
    result_naive = evalica.newman(xs, ys, ws, solver="naive")

    for result in (result_pyo3, result_naive):
        assert result.win_matrix.shape[0] == len(result.scores)
        assert np.isfinite(result.scores).all()
        assert np.isfinite(result.v)
        assert np.isfinite(result.v_init)
        assert result.iterations > 0

    tolerance = result_pyo3.tolerance * 10

    assert_series_equal(result_pyo3.scores, result_naive.scores, atol=tolerance)
    assert result_pyo3.v == pytest.approx(result_naive.v, abs=tolerance)


@given(example=elements())
def test_elo(example: Example) -> None:
    xs, ys, ws = example

    result = evalica.elo(xs, ys, ws)

    assert len(result.scores) == len(set(xs) | set(ys))
    assert np.isfinite(result.scores).all()


@given(example=elements())
def test_eigen(example: Example) -> None:
    xs, ys, ws = example

    result_pyo3 = evalica.eigen(xs, ys, ws, solver="pyo3")
    result_naive = evalica.eigen(xs, ys, ws, solver="naive")

    for result in (result_pyo3, result_naive):
        assert len(result.scores) == len(set(xs) | set(ys))
        assert np.isfinite(result.scores).all()
        assert result.iterations > 0

    tolerance = result_pyo3.tolerance * 10

    assert_series_equal(result_pyo3.scores, result_naive.scores, atol=tolerance)


@given(example=elements())
def test_pagerank(example: Example) -> None:
    xs, ys, ws = example

    result = evalica.pagerank(xs, ys, ws)

    assert len(result.scores) == len(set(xs) | set(ys))
    assert np.isfinite(result.scores).all()
    assert np.isfinite(result.damping)
    assert np.isfinite(result.win_weight)
    assert np.isfinite(result.tie_weight)
    assert not xs or result.iterations > 0


def test_bradley_terry_simple(simple_elements: Example) -> None:
    xs, ys, ws = simple_elements

    result_pyo3 = evalica.bradley_terry(xs, ys, ws, solver="pyo3")
    result_naive = evalica.bradley_terry(xs, ys, ws, solver="naive")

    tolerance = result_pyo3.tolerance * 10

    assert_series_equal(result_pyo3.scores, result_naive.scores, atol=tolerance)


def test_newman_simple(simple_tied_elements: Example) -> None:
    xs, ys, ws = simple_tied_elements

    result_pyo3 = evalica.newman(xs, ys, ws, solver="pyo3")
    result_naive = evalica.newman(xs, ys, ws, solver="naive")

    tolerance = result_pyo3.tolerance * 10

    assert_series_equal(result_pyo3.scores, result_naive.scores, atol=tolerance)
    assert result_pyo3.v == pytest.approx(result_naive.v, abs=tolerance)


@pytest.mark.parametrize(("algorithm", "dataset"), [
    ("counting", "food"),
    ("counting", "llmfao"),
])
def test_counting_dataset(example: Example, example_golden: "pd.Series[str]") -> None:
    xs, ys, ws = example

    result = evalica.counting(xs, ys, ws)

    assert_series_equal(result.scores, example_golden, check_like=True)


@pytest.mark.parametrize(("algorithm", "dataset"), [
    ("bradley_terry", "food"),
    ("bradley_terry", "llmfao"),
])
def test_bradley_terry_dataset(example: Example, example_golden: "pd.Series[str]") -> None:
    xs, ys, ws = example

    result_pyo3 = evalica.bradley_terry(xs, ys, ws, solver="pyo3")
    result_naive = evalica.bradley_terry(xs, ys, ws, solver="naive")

    tolerance = result_pyo3.tolerance * 10

    assert_series_equal(result_naive.scores, example_golden, atol=tolerance, check_like=True)
    assert_series_equal(result_pyo3.scores, example_golden, atol=tolerance, check_like=True)
    assert_series_equal(result_pyo3.scores, result_naive.scores, atol=tolerance, check_like=True)


@pytest.mark.parametrize(("algorithm", "dataset"), [
    ("newman", "food"),
    ("newman", "llmfao"),
])
def test_newman_dataset(example: Example, example_golden: "pd.Series[str]") -> None:
    xs, ys, ws = example

    result_pyo3 = evalica.newman(xs, ys, ws, solver="pyo3")
    result_naive = evalica.newman(xs, ys, ws, solver="naive")

    tolerance = result_pyo3.tolerance * 10

    assert_series_equal(result_naive.scores, example_golden, atol=tolerance, check_like=True)
    assert_series_equal(result_pyo3.scores, example_golden, atol=tolerance, check_like=True)

    assert_series_equal(result_pyo3.scores, result_naive.scores, atol=tolerance, check_like=True)
    assert result_pyo3.v == pytest.approx(result_naive.v, abs=tolerance)


@pytest.mark.parametrize(("algorithm", "dataset"), [
    ("elo", "food"),
    ("elo", "llmfao"),
])
def test_elo_dataset(example: Example, example_golden: "pd.Series[str]") -> None:
    xs, ys, ws = example

    result = evalica.elo(xs, ys, ws, initial=1000, k=4, scale=400)

    assert_series_equal(result.scores, example_golden, check_like=True)


@pytest.mark.parametrize(("algorithm", "dataset"), [
    ("eigen", "food"),
    ("eigen", "llmfao"),
])
def test_eigen_dataset(example: Example, example_golden: "pd.Series[str]") -> None:
    xs, ys, ws = example

    result_pyo3 = evalica.eigen(xs, ys, ws, solver="pyo3")
    result_naive = evalica.eigen(xs, ys, ws, solver="naive")

    tolerance = result_pyo3.tolerance * 10

    assert_series_equal(result_naive.scores, example_golden, atol=tolerance, check_like=True)
    assert_series_equal(result_pyo3.scores, example_golden, atol=tolerance, check_like=True)
    assert_series_equal(result_pyo3.scores, result_naive.scores, atol=tolerance, check_like=True)


@pytest.mark.parametrize(("algorithm", "dataset"), [
    ("pagerank", "food"),
    ("pagerank", "llmfao"),
])
def test_pagerank_dataset(example: Example, example_golden: "pd.Series[str]") -> None:
    xs, ys, ws = example

    result = evalica.pagerank(xs, ys, ws)

    tolerance = result.tolerance * 10

    assert_series_equal(result.scores, example_golden, atol=tolerance, check_like=True)
