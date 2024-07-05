#!/usr/bin/env python3

"""
A brute-force implementation of the Bradley-Terry ranking aggregation algorithm from the paper
MM algorithms for generalized Bradley-Terry models
<https://doi.org/10.1214/aos/1079120141>.
"""

__author__ = 'Dmitry Ustalov'
__copyright__ = 'Copyright 2021 Dmitry Ustalov'
__license__ = 'MIT'  # https://opensource.org/licenses/MIT

import numpy as np
import numpy.typing as npt
from scipy.optimize import fmin_cg

# M_ij indicates the number of times item 'i' ranks higher than item 'j'
M: npt.NDArray[np.int64] = np.array([
    [0, 1, 2, 0, 1],
    [2, 0, 2, 1, 0],
    [1, 2, 0, 0, 1],
    [1, 2, 1, 0, 2],
    [2, 0, 1, 3, 0],
], dtype=np.int64)


def loss(p: npt.NDArray[np.float64], M: npt.NDArray[np.int64]) -> np.float64:
    P = np.broadcast_to(p, M.shape).T  # whole i-th row should be p_i

    pi: np.float64 = (M * np.log(P)).sum()
    pij: np.float64 = (M * np.log(P + P.T)).sum()

    return pij - pi  # likelihood is (pi - pij) and we need a loss function


def main() -> None:
    p = fmin_cg(loss, np.ones(M.shape[0]), args=(M,))
    p /= p.sum()
    print(p)  # close to [0.12151104 0.15699947 0.11594851 0.31022851 0.29531247]
    print(p[:, np.newaxis] / (p + p[:, np.newaxis]))


if __name__ == '__main__':
    main()
