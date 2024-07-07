import numpy as np
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
def test_enumerate_elements(example: Example) -> None:  # type: ignore[type-var]
    xs, ys, ws = example

    index = evalica.enumerate_elements(xs, ys)

    assert isinstance(index, dict)
    assert len(index) == len(set(xs) | set(ys))
    assert not xs or max(index.values()) == len(index) - 1


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

    result = evalica.bradley_terry(xs, ys, ws)

    assert result.matrix.shape[0] == len(result.scores)
    assert np.isfinite(result.scores).all()
    assert result.iterations > 0


@given(example=elements())
def test_newman(example: Example) -> None:
    xs, ys, ws = example

    result = evalica.newman(xs, ys, ws)

    assert result.win_matrix.shape[0] == len(result.scores)
    assert np.isfinite(result.scores).all()
    assert np.isfinite(result.v)
    assert np.isfinite(result.v_init)
    assert result.iterations > 0


@given(example=elements())
def test_elo(example: Example) -> None:
    xs, ys, ws = example

    result = evalica.elo(xs, ys, ws)

    assert len(result.scores) == len(set(xs) | set(ys))
    assert np.isfinite(result.scores).all()


@given(example=elements())
def test_eigen(example: Example) -> None:
    xs, ys, ws = example

    result = evalica.eigen(xs, ys, ws)

    assert len(result.scores) == len(set(xs) | set(ys))
    assert np.isfinite(result.scores).all()


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


def test_bradley_terry_simple(simple_elements: Example, tolerance: float = 1e-4) -> None:
    xs, ys, ws = simple_elements

    result_pyo3 = evalica.bradley_terry(xs, ys, ws, solver="pyo3", tolerance=tolerance)
    result_naive = evalica.bradley_terry(xs, ys, ws, solver="naive", tolerance=tolerance)

    assert_series_equal(result_pyo3.scores, result_naive.scores, atol=tolerance)


def test_newman_simple(simple_tied_elements: Example, tolerance: float = 1e-1) -> None:
    xs, ys, ws = simple_tied_elements

    result_pyo3 = evalica.newman(xs, ys, ws, solver="pyo3", tolerance=tolerance)
    result_naive = evalica.newman(xs, ys, ws, solver="naive", tolerance=tolerance)

    assert_series_equal(result_pyo3.scores, result_naive.scores, atol=tolerance)


def test_bradley_terry_food(food: Example) -> None:
    xs, ys, ws = food

    result = evalica.bradley_terry(xs, ys, ws)

    assert len(result.scores) == len(set(xs) | set(ys))
    assert np.isfinite(result.scores).all()
    assert result.iterations > 0


def test_newman_food(food: Example) -> None:
    xs, ys, ws = food

    result = evalica.newman(xs, ys, ws)

    assert len(result.scores) == len(set(xs) | set(ys))
    assert np.isfinite(result.scores).all()
    assert np.isfinite(result.v)
    assert np.isfinite(result.v_init)
    assert result.iterations > 0


def test_elo_food(food: Example) -> None:
    xs, ys, ws = food

    result = evalica.elo(xs, ys, ws)

    assert len(result.scores) == len(set(xs) | set(ys))
    assert np.isfinite(result.scores).all()


def test_bradley_terry_llmfao(llmfao: Example) -> None:
    xs, ys, ws = llmfao

    result = evalica.bradley_terry(xs, ys, ws)

    assert len(result.scores) == len(set(xs) | set(ys))
    assert np.isfinite(result.scores).all()
    assert result.iterations > 0


def test_newman_llmfao(llmfao: Example) -> None:
    xs, ys, ws = llmfao

    result = evalica.newman(xs, ys, ws)

    assert len(result.scores) == len(set(xs) | set(ys))
    assert np.isfinite(result.scores).all()
    assert np.isfinite(result.v)
    assert np.isfinite(result.v_init)
    assert result.iterations > 0


def test_elo_llmfao(llmfao: Example) -> None:
    xs, ys, ws = llmfao

    result = evalica.elo(xs, ys, ws)

    assert len(result.scores) == len(set(xs) | set(ys))
    assert np.isfinite(result.scores).all()
