import numpy as np
import numpy.typing as npt
from hypothesis import given
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays

import evalica


def test_version() -> None:
    assert isinstance(evalica.__version__, str)
    assert len(evalica.__version__) > 0

@given(
    st.lists(st.integers(0, 2), min_size=2, max_size=2),
    st.lists(st.integers(0, 2), min_size=2, max_size=2),
    st.lists(st.integers(0, 3), min_size=2, max_size=2),
)
def test_matrices(
    first: list[int], second: list[int], statuses: list[int]
) -> None:
    n = 1 + max(max(first), max(second))

    win_count = sum(0 <= status <= 1 for status in statuses)
    tie_count = sum(status == 2 for status in statuses)

    wins, ties = evalica.matrices(first, second, statuses)

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

@given(arrays(dtype=np.int64, shape=(5, 5), elements=st.integers(0, 256)))
def test_newman(m: npt.NDArray[np.int64]) -> None:
    p, iterations = evalica.newman(m, 0, 1e-4, 100)

    assert m.shape[0] == len(p)
    assert np.isfinite(p).all()
    assert iterations > 0

@given(
    st.lists(st.integers(0, 2), min_size=2, max_size=2),
    st.lists(st.integers(0, 2), min_size=2, max_size=2),
    st.lists(st.integers(0, 3), min_size=2, max_size=2),
)
def test_elo(
    first: list[int], second: list[int], statuses: list[int]
) -> None:
    n = 1 + max(max(first), max(second))
    p = evalica.elo(first, second, statuses, 1500, 30, 400)

    assert n == len(p)
    assert np.isfinite(p).all()
