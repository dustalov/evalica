from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
import numpy.typing as npt

from evalica import LengthMismatchError, Winner

if TYPE_CHECKING:
    from collections.abc import Collection

    S = TypeVar("S", bound=npt.NBitBase)
    T = TypeVar("T")


def pairwise_scores(scores: npt.NDArray[np.number[S]]) -> npt.NDArray[np.float64]:
    if not scores.size:
        return np.zeros((0, 0))

    matrix = np.nan_to_num(
        scores[:, np.newaxis] / (scores + scores[:, np.newaxis]),
        copy=False,
    ).astype(np.float64, copy=False)

    return cast("npt.NDArray[np.float64]", matrix)


def _check_lengths(xs: Collection[Any], *rest: Collection[Any]) -> None:
    length = len(xs)

    for collection in rest:
        if len(collection) != length:
            raise LengthMismatchError


def counting(
        xs: Collection[int],
        ys: Collection[int],
        winners: Collection[Winner],
        weights: Collection[float],
        total: int,
        win_weight: float,
        tie_weight: float,
) -> npt.NDArray[np.float64]:
    _check_lengths(xs, ys, winners, weights)

    if not xs:
        return np.zeros(0)

    scores = np.zeros(total)

    with np.errstate(all="ignore"):
        for x, y, w, weight in zip(xs, ys, winners, weights):
            if w == Winner.X:
                scores[x] += weight * win_weight
            elif w == Winner.Y:
                scores[y] += weight * win_weight
            elif w == Winner.Draw:
                scores[x] += weight * tie_weight
                scores[y] += weight * tie_weight

    return np.nan_to_num(scores, copy=False)


def bradley_terry(
        matrix: npt.NDArray[np.number[S]],
        tolerance: float = 1e-6,
        limit: int = 100,
) -> tuple[npt.NDArray[np.float64], int]:
    scores = np.ones(matrix.shape[0])

    converged, iterations = False, 0

    if not matrix.size:
        return scores, iterations

    scores_new = scores.copy()

    while not converged and iterations < limit:
        iterations += 1

        with np.errstate(all="ignore"):
            for i in range(matrix.shape[0]):
                sums = scores_new[i] + scores_new

                numerator = np.sum(matrix[i] * scores_new / sums)
                denominator = np.sum(matrix[:, i] / sums)

                scores_new[i] = numerator / denominator

            geometric_mean = np.exp(np.mean(np.log(scores_new)))

            scores_new /= geometric_mean

        scores_new[:] = np.nan_to_num(scores_new, nan=tolerance, copy=False)

        with np.errstate(all="ignore"):
            converged = bool(np.linalg.norm(scores_new - scores) < tolerance)

        scores[:] = scores_new

    return scores, iterations


def newman(
        win_matrix: npt.NDArray[np.number[S]],
        tie_matrix: npt.NDArray[np.number[S]],
        v: float = .5,
        tolerance: float = 1e-6,
        limit: int = 100,
) -> tuple[npt.NDArray[np.float64], float, int]:
    with np.errstate(all="ignore"):
        win_tie_half = np.nan_to_num(win_matrix + tie_matrix / 2, nan=tolerance, copy=False)

    scores = np.ones(win_matrix.shape[0])

    converged, iterations = False, 0

    if not win_matrix.size and not tie_matrix.size:
        return scores, v, iterations

    scores_new = scores.copy()
    v_new = v

    while not converged and iterations < limit:
        iterations += 1

        v = np.nan_to_num(v_new, nan=tolerance)

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
            scores_new[:] = np.nan_to_num(scores_numerator / scores_denominator, nan=tolerance, copy=False)

        with np.errstate(all="ignore"):
            v_numerator = np.sum(tie_matrix * sum_scores / common_denominator) / 2
            v_denominator = np.sum(win_matrix * sqrt_scores_outer / common_denominator) * 2
            v_new = v_numerator / v_denominator

            converged = bool(np.linalg.norm(scores_new - scores) < tolerance)

        scores[:] = scores_new

    return scores, v, iterations


def elo(
        xs: Collection[int],
        ys: Collection[int],
        winners: Collection[Winner],
        weights: Collection[float],
        total: int,
        initial: float = 1000.,
        base: float = 10.,
        scale: float = 400.,
        k: float = 30.,
        win_weight: float = 1.0,
        tie_weight: float = 0.5,
) -> npt.NDArray[np.float64]:
    _check_lengths(xs, ys, winners, weights)

    if not xs:
        return np.zeros(0)

    scores = np.ones(total) * initial

    with np.errstate(all="ignore"):
        for x, y, w, weight in zip(xs, ys, winners, weights):
            q_x = np.nan_to_num(np.power(base, scores[x] / scale))
            q_y = np.nan_to_num(np.power(base, scores[y] / scale))

            q = np.nan_to_num(q_x + q_y)

            expected_x = np.nan_to_num(q_x / q)
            expected_y = np.nan_to_num(q_y / q)

            scored_x, scored_y = 0., 0.

            if w == Winner.X:
                scored_x = weight * win_weight
            elif w == Winner.Y:
                scored_y = weight * win_weight
            elif w == Winner.Draw:
                scored_x = scored_y = weight * tie_weight

            scores[x] += k * (scored_x - expected_x)
            scores[y] += k * (scored_y - expected_y)

    return np.nan_to_num(scores, copy=False)


def eigen(
        matrix: npt.NDArray[np.number[S]],
        tolerance: float = 1e-6,
        limit: int = 100,
) -> tuple[npt.NDArray[np.float64], int]:
    if not matrix.size:
        return np.zeros(0), 0

    n = matrix.shape[0]

    scores = np.ones(n) / n
    scores_new = scores.copy()

    converged, iterations = False, 0

    while not converged and iterations < limit:
        iterations += 1

        scores_new[:] = matrix.T @ scores

        with np.errstate(all="ignore"):
            scores_new /= np.linalg.norm(scores_new) or 1

        scores_new[:] = np.nan_to_num(scores_new, nan=tolerance, copy=False)

        with np.errstate(all="ignore"):
            converged = bool(np.linalg.norm(scores_new - scores) < tolerance)

        scores[:] = scores_new

    return scores.astype(np.float64, copy=False), iterations


def pagerank_matrix(
        matrix: npt.NDArray[np.number[S]],
        damping: float,
) -> npt.NDArray[np.float64]:
    if not matrix.size:
        return np.zeros(0)

    p = 1 / int(matrix.shape[0])

    _matrix = matrix.astype(np.float64, copy=False).T

    with np.errstate(all="ignore"):
        _matrix[_matrix.sum(axis=1) == 0] = p
        _matrix /= _matrix.sum(axis=1).reshape(-1, 1)
        _matrix = np.nan_to_num(_matrix, copy=False)

    return (damping * _matrix + (1 - damping) * p).astype(np.float64, copy=False)



def pagerank(
        matrix: npt.NDArray[np.number[S]],
        damping: float,
        tolerance: float,
        limit: int,
) -> tuple[npt.NDArray[np.float64], int]:
    _matrix = pagerank_matrix(matrix, damping)

    scores, iterations = eigen(_matrix, tolerance=tolerance, limit=limit)
    scores /= np.linalg.norm(scores, ord=1)

    return scores, iterations
