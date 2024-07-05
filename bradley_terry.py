#!/usr/bin/env python3

"""
An implementation of the Bradley-Terry ranking aggregation algorithm from the paper
MM algorithms for generalized Bradley-Terry models
<https://doi.org/10.1214/aos/1079120141>.
"""

__author__ = 'Dmitry Ustalov'
__copyright__ = 'Copyright 2021 Dmitry Ustalov'
__license__ = 'MIT'  # https://opensource.org/licenses/MIT

import numpy as np
import numpy.typing as npt

EPS = 1e-8

# M_ij indicates the number of times item 'i' ranks higher than item 'j'
M: npt.NDArray[np.int64] = np.array([
    [0, 1, 2, 0, 1],
    [2, 0, 2, 1, 0],
    [1, 2, 0, 0, 1],
    [1, 2, 1, 0, 2],
    [2, 0, 1, 3, 0],
], dtype=np.int64)


def main() -> None:
    T = M.T + M
    active = T > 0

    w = M.sum(axis=1)

    Z = np.zeros_like(M, dtype=float)

    p = np.ones(M.shape[0])
    p_new = p.copy()

    converged, iterations = False, 0

    while not converged:
        iterations += 1

        P = np.broadcast_to(p, M.shape)

        Z[active] = T[active] / (P[active] + P.T[active])

        p_new[:] = w
        p_new /= Z.sum(axis=0)
        p_new /= p_new.sum()

        converged = bool(np.linalg.norm(p_new - p) < EPS)

        p[:] = p_new

    print(f'{iterations} iteration(s)')
    print(p)  # [0.12151104 0.15699947 0.11594851 0.31022851 0.29531247]
    print(p[:, np.newaxis] / (p + p[:, np.newaxis]))


if __name__ == '__main__':
    main()
