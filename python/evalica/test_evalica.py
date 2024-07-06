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
    st.lists(st.integers(0, 2), min_size=2, max_size=2),
    st.lists(st.sampled_from(evalica.WINNERS), min_size=2, max_size=2),
)
def test_matrices(
        xs: list[int], ys: list[int], ws: list[evalica.Winner],
) -> None:
    n = 1 + max(max(xs), max(ys))

    win_count = sum(status in [evalica.Winner.X, evalica.Winner.Y] for status in ws)
    tie_count = sum(status == evalica.Winner.Draw for status in ws)

    wins, ties = evalica.matrices(xs, ys, ws)

    assert wins.shape == (n, n)
    assert ties.shape == (n, n)
    assert wins.sum() == win_count
    assert ties.sum() == 2 * tie_count


@given(arrays(dtype=np.int64, shape=(5, 5), elements=st.integers(0, 256)))
def test_counting(m: npt.NDArray[np.int64]) -> None:
    p = evalica.counting(m)

    assert m.shape[0] == len(p)
    assert np.isfinite(p).all()


@given(arrays(dtype=np.float64, shape=(5, 5), elements=st.integers(0, 256)))
def test_bradley_terry(m: npt.NDArray[np.float64]) -> None:
    p, iterations = evalica.bradley_terry(m, 1e-4, 100)

    assert m.shape[0] == len(p)
    assert np.isfinite(p).all()
    assert iterations > 0


@given(
    arrays(dtype=np.float64, shape=(5, 5), elements=st.integers(0, 256)),
    arrays(dtype=np.float64, shape=(5, 5), elements=st.integers(0, 256)),
    st.floats(allow_nan=False, allow_infinity=False, allow_subnormal=False,
              min_value=0., max_value=1., exclude_min=True),
)
def test_newman(w: npt.NDArray[np.float64], t: npt.NDArray[np.float64], v_init: float) -> None:
    p, v, iterations = evalica.newman(w, t, v_init, 1e-4, 100)

    assert w.shape[0] == len(p)
    assert np.isfinite(p).all()
    assert np.isfinite(v)
    assert iterations > 0


@given(
    st.lists(st.integers(0, 2), min_size=2, max_size=2),
    st.lists(st.integers(0, 2), min_size=2, max_size=2),
    st.lists(st.sampled_from(evalica.WINNERS), min_size=2, max_size=2),
)
def test_elo(
        xs: list[int], ys: list[int], ws: list[evalica.Winner],
) -> None:
    n = 1 + max(max(xs), max(ys))
    p = evalica.elo(xs, ys, ws, 1500, 30, 400)

    assert n == len(p)
    assert np.isfinite(p).all()


@given(arrays(dtype=np.int64, shape=(5, 5), elements=st.integers(0, 256)))
def test_eigen(m: npt.NDArray[np.int64]) -> None:
    p = evalica.eigen(m.astype(np.float64))

    assert m.shape[0] == len(p)
    assert np.isfinite(p).all()


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
    p, _ = evalica.bradley_terry(simple, tolerance, 100)

    assert p == pytest.approx(p_naive, abs=tolerance)


def test_newman_simple(simple_win_tie: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
                       tolerance: float = 1e-1) -> None:
    w, t = simple_win_tie

    p_naive, _, _ = evalica.newman_naive(w, t, .5, tolerance)
    p, _, _ = evalica.newman(w, t, .5, tolerance, 100)

    assert p == pytest.approx(p_naive, abs=tolerance)


@pytest.fixture()
def food() -> tuple[list[int], list[int], list[evalica.Winner]]:
    df_food = pd.read_csv(Path().parent.parent / "food.csv", dtype=str)

    xs = df_food["left"]
    ys = df_food["right"]
    ws = df_food["winner"].map({
        "left": Winner.X,
        "right": Winner.Y,
        "tie": Winner.Draw,
    })

    # TODO: we need a better DX than this
    index: dict[str, int] = {}

    for xy in zip(xs, ys, strict=False):
        for e in xy:
            index[e] = index.get(e, len(index))

    return [index[x] for x in xs], [index[y] for y in ys], ws.tolist()


def test_bradley_terry_food(food: tuple[list[int], list[int], list[evalica.Winner]]) -> None:
    xs, ys, ws = food

    _wins, _ties = evalica.matrices(xs, ys, ws)
    wins = _wins.astype(np.float64) + _ties / 2

    scores, iterations = evalica.bradley_terry(wins, 1e-4, 100)

    assert len(set(xs) | set(ys)) == len(scores)
    assert np.isfinite(scores).all()
    assert iterations > 0


def test_elo_food(food: tuple[list[int], list[int], list[evalica.Winner]]) -> None:
    xs, ys, ws = food

    scores = evalica.elo(xs, ys, ws, 1500, 30, 400)

    assert len(scores) == len(set(xs) | set(ys))
    assert np.isfinite(scores).all()
