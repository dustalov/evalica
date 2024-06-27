#!/usr/bin/env python3

import unittest

import numpy as np
import numpy.typing as npt
from hypothesis import given
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays

import evalica


class TestMeta(unittest.TestCase):
    def test_version(self) -> None:
        self.assertIsInstance(evalica.__version__, str)
        self.assertGreater(len(evalica.__version__), 0)


class TestUnordered(unittest.TestCase):
    def setUp(self) -> None:
        self.M: npt.NDArray[np.int64] = np.array(
            [
                [0, 1, 2, 0, 1],
                [2, 0, 2, 1, 0],
                [1, 2, 0, 0, 1],
                [1, 2, 1, 0, 2],
                [2, 0, 1, 3, 0],
            ],
            dtype=np.int64,
        )

    @given(
        st.lists(st.integers(0, 2), min_size=2, max_size=2),
        st.lists(st.integers(0, 2), min_size=2, max_size=2),
        st.lists(st.integers(0, 3), min_size=2, max_size=2),
    )
    def test_matrices(
        self, first: list[int], second: list[int], statuses: list[int]
    ) -> None:
        n = 1 + max(max(first), max(second))

        win_count = sum(0 <= status <= 1 for status in statuses)
        tie_count = sum(status == 2 for status in statuses)

        wins, ties = evalica.matrices(first, second, statuses)

        self.assertEqual(wins.shape, (n, n))
        self.assertEqual(ties.shape, (n, n))
        self.assertEqual(wins.sum(), win_count)
        self.assertEqual(ties.sum(), 2 * tie_count)

    @given(arrays(dtype=np.int64, shape=(5, 5), elements=st.integers(0, 256)))
    def test_counting(self, m: npt.NDArray[np.int64]) -> None:
        p = evalica.counting(m)

        self.assertTrue(np.isfinite(p).all())

    @given(arrays(dtype=np.int64, shape=(5, 5), elements=st.integers(0, 256)))
    def test_bradley_terry(self, m: npt.NDArray[np.int64]) -> None:
        p, iterations = evalica.bradley_terry(self.M, 1e-4, 100)

        self.assertTrue(np.isfinite(p).all())
        self.assertGreater(iterations, 0)

    @given(arrays(dtype=np.int64, shape=(5, 5), elements=st.integers(0, 256)))
    def test_newman(self, m: npt.NDArray[np.int64]) -> None:
        p, iterations = evalica.newman(m, 0, 1e-4, 100)

        self.assertTrue(np.isfinite(p).all())
        self.assertGreater(iterations, 0)


if __name__ == "__main__":
    unittest.main()
