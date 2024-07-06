import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays

import evalica


def test_version() -> None:
    assert isinstance(evalica.__version__, str)
    assert len(evalica.__version__) > 0


def test_exports() -> None:
    for attr in evalica.__all__:
        assert hasattr(evalica, attr), f"missing attribute: {attr}"


@given(
    st.lists(st.integers(0, 2), min_size=2, max_size=2),
    st.lists(st.integers(0, 2), min_size=2, max_size=2),
    st.lists(st.sampled_from(evalica.STATUSES), min_size=2, max_size=2),
)
def test_matrices(
        xs: list[int], ys: list[int], rs: list[evalica.Status],
) -> None:
    n = 1 + max(max(xs), max(ys))

    win_count = sum(status in [evalica.Status.Won, evalica.Status.Lost] for status in rs)
    tie_count = sum(status == evalica.Status.Tied for status in rs)

    wins, ties = evalica.matrices(xs, ys, rs)

    assert wins.shape == (n, n)
    assert ties.shape == (n, n)
    assert wins.sum() == win_count
    assert ties.sum() == 2 * tie_count


@given(arrays(dtype=np.int64, shape=(5, 5), elements=st.integers(0, 256)))
def test_counting(m: npt.NDArray[np.int64]) -> None:
    p = evalica.counting(m)

    assert m.shape[0] == len(p)
    assert np.isfinite(p).all()


@given(arrays(dtype=np.int64, shape=(5, 5), elements=st.integers(0, 256)))
def test_bradley_terry(m: npt.NDArray[np.int64]) -> None:
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
    st.lists(st.sampled_from(evalica.STATUSES), min_size=2, max_size=2),
)
def test_elo(
        xs: list[int], ys: list[int], rs: list[evalica.Status],
) -> None:
    n = 1 + max(max(xs), max(ys))
    p = evalica.elo(xs, ys, rs, 1500, 30, 400)

    assert n == len(p)
    assert np.isfinite(p).all()


@given(arrays(dtype=np.int64, shape=(5, 5), elements=st.integers(0, 256)))
def test_eigen(m: npt.NDArray[np.int64]) -> None:
    p = evalica.eigen(m.astype(np.float64))

    assert m.shape[0] == len(p)
    assert np.isfinite(p).all()


@pytest.fixture()
def simple() -> npt.NDArray[np.int64]:
    return np.array([
        [0, 1, 2, 0, 1],
        [2, 0, 2, 1, 0],
        [1, 2, 0, 0, 1],
        [1, 2, 1, 0, 2],
        [2, 0, 1, 3, 0],
    ], dtype=np.int64)


@pytest.fixture()
def simple_win_tie(simple: npt.NDArray[np.int64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    T = np.minimum(simple, simple.T).astype(np.float64)  # noqa: N806
    W = simple - T  # noqa: N806

    return W, T


def test_bradley_terry_simple(simple: npt.NDArray[np.int64], tolerance: float = 1e-4) -> None:
    p_naive, _ = evalica.bradley_terry_naive(simple, tolerance)
    p, _ = evalica.bradley_terry(simple, tolerance, 100)

    assert p == pytest.approx(p_naive, abs=tolerance)


def test_newman_simple(simple_win_tie: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
                       tolerance: float = 1e-1) -> None:
    w, t = simple_win_tie

    p_naive, _, _ = evalica.newman_naive(w, t, .5, tolerance)
    p, _, _ = evalica.newman(w, t, .5, tolerance, 100)

    # TODO: they are diverging
    assert p == pytest.approx(p_naive, abs=tolerance)
