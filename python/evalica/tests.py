#!/usr/bin/env python3

import unittest

import numpy as np
import numpy.typing as npt

import evalica


class TestUnordered(unittest.TestCase):
    @classmethod
    def setUp(self) -> None:
        self.M: npt.NDArray[np.int64] = np.array([
            [0, 1, 2, 0, 1],
            [2, 0, 2, 1, 0],
            [1, 2, 0, 0, 1],
            [1, 2, 1, 0, 2],
            [2, 0, 1, 3, 0],
        ], dtype=np.int64)

    def test_bradley_terry(self) -> None:
        p, iterations = evalica.bradley_terry(self.M)

        assert np.isfinite(p).all()
        assert iterations > 0

    def test_newman(self) -> None:
        p, iterations = evalica.newman(self.M, 0, 1e-6, 100)

        assert np.isfinite(p).all()
        assert iterations > 0


if __name__ == '__main__':
    unittest.main()
