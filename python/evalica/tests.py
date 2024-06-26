#!/usr/bin/env python3

import unittest

import numpy as np
import numpy.typing as npt

import evalica


class TestMeta(unittest.TestCase):
    def test_version(self) -> None:
        self.assertIsInstance(evalica.__version__, str)
        self.assertGreater(len(evalica.__version__), 0)


class TestUnordered(unittest.TestCase):
    def setUp(self) -> None:
        self.M: npt.NDArray[np.int64] = np.array([
            [0, 1, 2, 0, 1],
            [2, 0, 2, 1, 0],
            [1, 2, 0, 0, 1],
            [1, 2, 1, 0, 2],
            [2, 0, 1, 3, 0],
        ], dtype=np.int64)

    def test_matrices(self) -> None:
        first = [0, 1]
        second = [1, 0]
        statuses = [0, 0]

        wins, ties = evalica.matrices(first, second, statuses)

        self.assertEqual(wins.shape, (2, 2))
        self.assertEqual(ties.shape, (2, 2))
        self.assertSequenceEqual(wins.tolist(), [[0, 1], [1, 0]])
        self.assertSequenceEqual(ties.tolist(), [[0, 0], [0, 0]])

    def test_counting(self) -> None:
        p = evalica.counting(self.M)

        self.assertTrue(np.isfinite(p).all())

    def test_bradley_terry(self) -> None:
        p, iterations = evalica.bradley_terry(self.M)

        self.assertTrue(np.isfinite(p).all())
        self.assertGreater(iterations, 0)

    def test_newman(self) -> None:
        p, iterations = evalica.newman(self.M, 0, 1e-6, 100)

        self.assertTrue(np.isfinite(p).all())
        self.assertGreater(iterations, 0)


if __name__ == '__main__':
    unittest.main()
