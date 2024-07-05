#!/usr/bin/env python3

"""
An implementation of the ranking aggregation algorithm from the paper
Efficient Computation of Rankings from Pairwise Comparisons
<https://www.jmlr.org/papers/v24/22-1086.html>.
"""

__author__ = 'Dmitry Ustalov'
__copyright__ = 'Copyright 2023 Dmitry Ustalov'
__license__ = 'MIT'  # https://opensource.org/licenses/MIT

import numpy as np
import numpy.typing as npt

# M_ij indicates the number of times item 'i' ranks higher than item 'j'
M: npt.NDArray[np.int64] = np.array([
    [0, 1, 2, 0, 1],
    [2, 0, 2, 1, 0],
    [1, 2, 0, 0, 1],
    [1, 2, 1, 0, 2],
    [2, 0, 1, 3, 0],
], dtype=np.int64)

# tie matrix
T = np.minimum(M, M.T)

# win matrix
W = M - T

EPS = 10e-6


def main() -> None:
    np.random.seed(0)

    pi = np.random.rand(M.shape[0])
    v = np.random.rand()

    converged, iterations = False, 0

    while not converged:
        iterations += 1

        v_numerator = np.sum(
            T * (pi[:, np.newaxis] + pi) /
            (pi[:, np.newaxis] + pi + 2 * v * np.sqrt(pi[:, np.newaxis] * pi))
        ) / 2

        v_denominator = np.sum(
            W * 2 * np.sqrt(pi[:, np.newaxis] * pi) /
            (pi[:, np.newaxis] + pi + 2 * v * np.sqrt(pi[:, np.newaxis] * pi))
        )

        v = v_numerator / v_denominator

        pi_old = pi.copy()

        pi_numerator = np.sum(
            (W + T / 2) * (pi + v * np.sqrt(pi[:, np.newaxis] * pi)) /
            (pi[:, np.newaxis] + pi + 2 * v * np.sqrt(pi[:, np.newaxis] * pi)),
            axis=1
        )

        pi_denominator = np.sum(
            (W + T / 2) * (1 + v * np.sqrt(pi[:, np.newaxis] * pi)) /
            (pi[:, np.newaxis] + pi + 2 * v * np.sqrt(pi[:, np.newaxis] * pi)),
            axis=0
        )

        pi = pi_numerator / pi_denominator

        converged = np.allclose(pi / (pi + 1), pi_old / (pi_old + 1), rtol=EPS, atol=EPS)

    print(f'{iterations} iteration(s)')
    print(pi)
    print(pi[:, np.newaxis] / (pi + pi[:, np.newaxis]))


if __name__ == '__main__':
    main()
