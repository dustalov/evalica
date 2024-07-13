from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from evalica import Winner

if TYPE_CHECKING:
    from collections.abc import Collection


def bradley_terry(
        matrix: npt.NDArray[np.float64],
        tolerance: float = 1e-6,
        limit: int = 100,
) -> tuple[npt.NDArray[np.float64], int]:
    totals = matrix.T + matrix
    active = totals > 0

    wins = matrix.sum(axis=1)

    norm_matrix = np.zeros_like(matrix, dtype=float)

    scores = np.ones(matrix.shape[0])
    scores_new = scores.copy()

    converged, iterations = False, 0

    while not converged and iterations < limit:
        iterations += 1

        broadcast_scores = np.broadcast_to(scores, matrix.shape)

        norm_matrix[active] = totals[active] / (broadcast_scores[active] + broadcast_scores.T[active])

        scores_new[:] = wins
        with np.errstate(invalid="ignore"):
            scores_new /= norm_matrix.sum(axis=0)
            scores_new /= scores_new.sum()
        scores_new = np.nan_to_num(scores_new, nan=tolerance)

        converged = bool(np.linalg.norm(scores_new - scores) < tolerance)

        scores[:] = np.nan_to_num(scores_new, nan=tolerance)

    return scores, iterations


def newman(
        win_matrix: npt.NDArray[np.float64],
        tie_matrix: npt.NDArray[np.float64],
        v: float = .5,
        tolerance: float = 1e-6,
        limit: int = 100,
) -> tuple[npt.NDArray[np.float64], float, int]:
    win_tie_half = win_matrix + tie_matrix / 2

    scores = np.ones(win_matrix.shape[0])
    scores_new = scores.copy()
    v_new = v

    converged, iterations = False, 0

    while not converged and iterations < limit:
        iterations += 1

        v = v_new

        broadcast_scores_t = scores[:, None].T

        with np.errstate(all="ignore"):
            sqrt_scores_outer = np.sqrt(np.outer(scores, scores))
            sum_scores = np.add.outer(scores, scores)
            sqrt_div_scores_outer_t = np.sqrt(np.divide.outer(scores, scores)).T
            common_denominator = sum_scores + 2 * v * sqrt_scores_outer

        with np.errstate(all="ignore"):
            scores_numerator = np.sum(
                win_tie_half * (broadcast_scores_t + v * sqrt_scores_outer) / common_denominator,
                axis=1,
            )
            scores_denominator = np.sum(
                win_tie_half.T * (1 + v * sqrt_div_scores_outer_t) / common_denominator,
                axis=1,
            )
            scores_new[:] = np.nan_to_num(scores_numerator / scores_denominator, nan=tolerance)

        with np.errstate(all="ignore"):
            v_numerator = np.sum(tie_matrix * sum_scores / common_denominator) / 2
            v_denominator = np.sum(win_matrix * sqrt_scores_outer / common_denominator) * 2
            v_new = np.nan_to_num(v_numerator / v_denominator, nan=tolerance)

        converged = bool(np.linalg.norm(scores_new - scores) < tolerance)

        scores[:] = scores_new

    return scores, v, iterations


def elo(
        xs: Collection[int],
        ys: Collection[int],
        ws: Collection[Winner],
        initial: float = 1000.,
        base: float = 10.,
        scale: float = 400.,
        k: float = 30.,
) -> npt.NDArray[np.float64]:
    if len(xs) != len(ys) or len(xs) != len(ws) or len(xs) != len(ws) or not xs:
        return np.zeros(0, dtype=np.float64)

    n = 1 + max(*xs, *ys)

    scores = np.ones(n) * initial

    for x, y, w in zip(xs, ys, ws):
        q_x = np.power(base, scores[x] / scale)
        q_y = np.power(base, scores[y] / scale)
        q = q_x + q_y

        expected_x, expected_y = q_x / q, q_y / q

        scored_x, scored_y = 0., 0.

        if w == Winner.X:
            scored_x, scored_y = 1., 0.
        elif w == Winner.Y:
            scored_x, scored_y = 0., 1.
        elif w == Winner.Draw:
            scored_x, scored_y = .5, .5

        scores[x] += k * (scored_x - expected_x)
        scores[y] += k * (scored_y - expected_y)

    return scores


def eigen(
        matrix: npt.NDArray[np.float64],
        tolerance: float = 1e-6,
        limit: int = 100,
) -> tuple[npt.NDArray[np.float64], int]:
    n = matrix.shape[0]

    scores = np.ones(n) / n
    scores_new = scores.copy()

    converged, iterations = False, 0

    while not converged and iterations < limit:
        iterations += 1

        scores_new[:] = matrix.T @ scores
        scores_new /= np.linalg.norm(scores_new) or 1
        scores_new[:] = np.nan_to_num(scores_new, nan=tolerance)

        converged = bool(np.linalg.norm(scores_new - scores) < tolerance)

        scores[:] = scores_new

    return scores, iterations
