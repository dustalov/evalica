import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given

import evalica
from conftest import Example, xs_ys_ws


def test_version() -> None:
    assert isinstance(evalica.__version__, str)
    assert len(evalica.__version__) > 0


def test_exports() -> None:
    for attr in evalica.__all__:
        assert hasattr(evalica, attr), f"missing attribute: {attr}"


@given(xs_ys_ws=xs_ys_ws())
def test_enumerate_elements(xs_ys_ws: Example) -> None:
    xs, ys, ws = xs_ys_ws

    index = evalica.enumerate_elements(xs, ys)

    assert isinstance(index, dict)
    assert len(index) == len(set(xs) | set(ys))
    assert not xs or max(index.values()) == len(index) - 1


@given(xs_ys_ws=xs_ys_ws())
def test_matrices(xs_ys_ws: Example) -> None:
    xs, ys, ws = xs_ys_ws

    n = len(set(xs) | set(ys))

    wins = sum(status in [evalica.Winner.X, evalica.Winner.Y] for status in ws)
    ties = sum(status == evalica.Winner.Draw for status in ws)

    result = evalica.matrices(xs, ys, ws)

    assert result.win_matrix.shape == (n, n)
    assert result.tie_matrix.shape == (n, n)
    assert result.win_matrix.sum() == wins
    assert result.tie_matrix.sum() == 2 * ties


@given(xs_ys_ws=xs_ys_ws())
def test_counting(xs_ys_ws: Example) -> None:
    xs, ys, ws = xs_ys_ws

    result = evalica.counting(xs, ys, ws)

    assert result.win_matrix.shape[0] == len(result.scores)
    assert np.isfinite(result.scores).all()


@given(xs_ys_ws=xs_ys_ws())
def test_bradley_terry(xs_ys_ws: Example) -> None:
    xs, ys, ws = xs_ys_ws

    result = evalica.bradley_terry(xs, ys, ws)

    assert result.matrix.shape[0] == len(result.scores)
    assert np.isfinite(result.scores).all()
    assert result.iterations > 0


@given(xs_ys_ws=xs_ys_ws())
def test_newman(xs_ys_ws: Example) -> None:
    xs, ys, ws = xs_ys_ws

    result = evalica.newman(xs, ys, ws)

    assert result.win_matrix.shape[0] == len(result.scores)
    assert np.isfinite(result.scores).all()
    assert np.isfinite(result.v)
    assert np.isfinite(result.v_init)
    assert result.iterations > 0


@given(xs_ys_ws=xs_ys_ws())
def test_elo(xs_ys_ws: Example) -> None:
    xs, ys, ws = xs_ys_ws

    result = evalica.elo(xs, ys, ws)

    assert len(result.scores) == len(set(xs) | set(ys))
    assert np.isfinite(result.scores).all()


@given(xs_ys_ws=xs_ys_ws())
def test_eigen(xs_ys_ws: Example) -> None:
    xs, ys, ws = xs_ys_ws

    result = evalica.eigen(xs, ys, ws)

    assert len(result.scores) == len(set(xs) | set(ys))
    assert np.isfinite(result.scores).all()


def test_bradley_terry_simple(simple: npt.NDArray[np.float64], tolerance: float = 1e-4) -> None:
    p_naive, _ = evalica.bradley_terry_naive(simple, tolerance)  # type: ignore[attr-defined]
    p, _ = evalica.bradley_terry_pyo3(simple, tolerance, 100)  # type: ignore[attr-defined]

    assert p == pytest.approx(p_naive, abs=tolerance)


def test_newman_simple(simple_win_tie: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
                       tolerance: float = 1e-1) -> None:
    w, t = simple_win_tie

    p_naive, _, _ = evalica.newman_naive(w, t, .5, tolerance)  # type: ignore[attr-defined]
    p, _, _ = evalica.newman_pyo3(w, t, .5, tolerance, 100)  # type: ignore[attr-defined]

    assert p == pytest.approx(p_naive, abs=tolerance)


def test_bradley_terry_food(food: tuple[list[str], list[str], list[evalica.Winner]]) -> None:
    xs, ys, ws = food

    result = evalica.bradley_terry(xs, ys, ws)

    assert len(result.scores) == len(set(xs) | set(ys))
    assert np.isfinite(result.scores).all()
    assert result.iterations > 0


def test_newman_food(food: tuple[list[str], list[str], list[evalica.Winner]]) -> None:
    xs, ys, ws = food

    result = evalica.newman(xs, ys, ws)

    assert len(result.scores) == len(set(xs) | set(ys))
    assert np.isfinite(result.scores).all()
    assert np.isfinite(result.v)
    assert np.isfinite(result.v_init)
    assert result.iterations > 0


def test_elo_food(food: tuple[list[str], list[str], list[evalica.Winner]]) -> None:
    xs, ys, ws = food

    result = evalica.elo(xs, ys, ws)

    assert len(result.scores) == len(set(xs) | set(ys))
    assert np.isfinite(result.scores).all()


def test_bradley_terry_llmfao(llmfao: tuple[list[str], list[str], list[evalica.Winner]]) -> None:
    xs, ys, ws = llmfao

    result = evalica.bradley_terry(xs, ys, ws)

    assert len(result.scores) == len(set(xs) | set(ys))
    assert np.isfinite(result.scores).all()
    assert result.iterations > 0


def test_newman_llmfao(llmfao: tuple[list[str], list[str], list[evalica.Winner]]) -> None:
    xs, ys, ws = llmfao

    result = evalica.newman(xs, ys, ws)

    assert len(result.scores) == len(set(xs) | set(ys))
    assert np.isfinite(result.scores).all()
    assert np.isfinite(result.v)
    assert np.isfinite(result.v_init)
    assert result.iterations > 0


def test_elo_llmfao(llmfao: tuple[list[str], list[str], list[evalica.Winner]]) -> None:
    xs, ys, ws = llmfao

    result = evalica.elo(xs, ys, ws)

    assert len(result.scores) == len(set(xs) | set(ys))
    assert np.isfinite(result.scores).all()
