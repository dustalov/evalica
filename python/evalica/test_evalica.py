from pathlib import Path

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays

import evalica
from evalica import Winner


def test_version() -> None:
    assert isinstance(evalica.__version__, str)
    assert len(evalica.__version__) > 0


def test_exports() -> None:
    for attr in evalica.__all__:
        assert hasattr(evalica, attr), f"missing attribute: {attr}"

@given(
    st.lists(st.integers(0, 2), min_size=2, max_size=2),
    arrays(dtype=np.int64, shape=(2,)),
)
def test_index(xs: list[int], ys: npt.NDArray[np.int64]) -> None:
    index = evalica.index(xs, ys)

    assert isinstance(index, dict)
    assert len(index) == len(set(xs) | set(ys))
    assert max(index.values()) == len(index) - 1


@given(
    st.lists(st.integers(0, 2), min_size=2, max_size=2),
    st.lists(st.integers(0, 2), min_size=2, max_size=2),
    st.lists(st.sampled_from(evalica.WINNERS), min_size=2, max_size=2),
)
def test_matrices(xs: list[int], ys: list[int], ws: list[evalica.Winner]) -> None:
    n = len(set(xs) | set(ys))

    wins = sum(status in [evalica.Winner.X, evalica.Winner.Y] for status in ws)
    ties = sum(status == evalica.Winner.Draw for status in ws)

    result = evalica.matrices(xs, ys, ws)

    assert result.win_matrix.shape == (n, n)
    assert result.tie_matrix.shape == (n, n)
    assert result.win_matrix.sum() == wins
    assert result.tie_matrix.sum() == 2 * ties


@given(
    st.lists(st.integers(0, 2), min_size=2, max_size=2),
    st.lists(st.integers(0, 2), min_size=2, max_size=2),
    st.lists(st.sampled_from(evalica.WINNERS), min_size=2, max_size=2),
)
def test_counting(xs: list[int], ys: list[int], ws: list[evalica.Winner]) -> None:
    result = evalica.counting(xs, ys, ws)

    assert result.win_matrix.shape[0] == len(result.scores)
    assert np.isfinite(result.scores).all()

@given(
    st.lists(st.integers(0, 2), min_size=2, max_size=2),
    st.lists(st.integers(0, 2), min_size=2, max_size=2),
    st.lists(st.sampled_from(evalica.WINNERS), min_size=2, max_size=2),
)
def test_bradley_terry(xs: list[int], ys: list[int], ws: list[evalica.Winner]) -> None:
    result = evalica.bradley_terry(xs, ys, ws)

    assert result.matrix.shape[0] == len(result.scores)
    assert np.isfinite(result.scores).all()
    assert result.iterations > 0


@given(
    st.lists(st.integers(0, 2), min_size=2, max_size=2),
    st.lists(st.integers(0, 2), min_size=2, max_size=2),
    st.lists(st.sampled_from(evalica.WINNERS), min_size=2, max_size=2),
)
def test_newman(xs: list[int], ys: list[int], ws: list[evalica.Winner]) -> None:
    result = evalica.newman(xs, ys, ws)

    assert result.win_matrix.shape[0] == len(result.scores)
    assert np.isfinite(result.scores).all()
    assert np.isfinite(result.v)
    assert np.isfinite(result.v_init)
    assert result.iterations > 0


@given(
    st.lists(st.integers(0, 2), min_size=2, max_size=2),
    st.lists(st.integers(0, 2), min_size=2, max_size=2),
    st.lists(st.sampled_from(evalica.WINNERS), min_size=2, max_size=2),
)
def test_elo(xs: list[int], ys: list[int], ws: list[evalica.Winner]) -> None:
    result = evalica.elo(xs, ys, ws)

    assert len(result.scores) == len(set(xs) | set(ys))
    assert np.isfinite(result.scores).all()


@given(
    st.lists(st.integers(0, 2), min_size=2, max_size=2),
    st.lists(st.integers(0, 2), min_size=2, max_size=2),
    st.lists(st.sampled_from(evalica.WINNERS), min_size=2, max_size=2),
)
def test_eigen(xs: list[int], ys: list[int], ws: list[evalica.Winner]) -> None:
    result = evalica.eigen(xs, ys, ws)

    assert len(result.scores) == len(set(xs) | set(ys))
    assert np.isfinite(result.scores).all()


@pytest.fixture()
def simple() -> npt.NDArray[np.float64]:
    return np.array([
        [0, 1, 2, 0, 1],
        [2, 0, 2, 1, 0],
        [1, 2, 0, 0, 1],
        [1, 2, 1, 0, 2],
        [2, 0, 1, 3, 0],
    ], dtype=np.float64)


@pytest.fixture()
def simple_win_tie(simple: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    T = np.minimum(simple, simple.T).astype(np.float64)  # noqa: N806
    W = simple - T  # noqa: N806

    return W, T


def test_bradley_terry_simple(simple: npt.NDArray[np.float64], tolerance: float = 1e-4) -> None:
    p_naive, _ = evalica.bradley_terry_naive(simple, tolerance)
    p, _ = evalica.py_bradley_terry(simple, tolerance, 100)  # type: ignore[attr-defined]

    assert p == pytest.approx(p_naive, abs=tolerance)


def test_newman_simple(simple_win_tie: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
                       tolerance: float = 1e-1) -> None:
    w, t = simple_win_tie

    p_naive, _, _ = evalica.newman_naive(w, t, .5, tolerance)
    p, _, _ = evalica.py_newman(w, t, .5, tolerance, 100)  # type: ignore[attr-defined]

    assert p == pytest.approx(p_naive, abs=tolerance)


@pytest.fixture()
def food() -> tuple[list[str], list[str], list[evalica.Winner]]:
    df_food = pd.read_csv(Path().parent.parent / "food.csv", dtype=str)

    xs = df_food["left"]
    ys = df_food["right"]
    ws = df_food["winner"].map({
        "left": Winner.X,
        "right": Winner.Y,
        "tie": Winner.Draw,
    })

    return xs.tolist(), ys.tolist(), ws.tolist()


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
