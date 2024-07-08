import numpy as np
import numpy.typing as npt


def bradley_terry(
        matrix: npt.NDArray[np.float64],
        tolerance: float = 1e-6,
        limit: int = 100,
) -> tuple[npt.NDArray[np.float64], int]:
    sum_matrix = matrix.T + matrix
    active = sum_matrix > 0

    w = matrix.sum(axis=1)

    norm_matrix = np.zeros_like(matrix, dtype=float)

    scores = np.ones(matrix.shape[0])
    scores_new = scores.copy()

    converged, iterations = False, 0

    while not converged and iterations < limit:
        iterations += 1

        broadcast_scores = np.broadcast_to(scores, matrix.shape)

        norm_matrix[active] = sum_matrix[active] / (broadcast_scores[active] + broadcast_scores.T[active])

        scores_new[:] = w
        scores_new /= norm_matrix.sum(axis=0)
        scores_new /= scores_new.sum()
        scores_new = np.nan_to_num(scores_new, nan=tolerance)

        converged = bool(np.linalg.norm(scores_new - scores) < tolerance)

        scores[:] = scores_new

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

    converged, iterations = False, 0

    while not converged:
        iterations += 1

        scores_broadcast = scores[:, np.newaxis]

        scores_outer_sqrt = np.sqrt(np.outer(scores, scores))

        v_numerator = np.sum(
            tie_matrix * (scores_broadcast + scores) /
            (scores_broadcast + scores + 2 * v * scores_outer_sqrt),
        ) / 2

        v_denominator = np.sum(
            win_matrix * 2 * scores_outer_sqrt /
            (scores_broadcast + scores + 2 * v * scores_outer_sqrt),
        )

        v = v_numerator / v_denominator
        v = np.nan_to_num(v, nan=tolerance)

        scores_old = scores.copy()

        pi_numerator = np.sum(
            win_tie_half * (scores + v * scores_outer_sqrt) /
            (scores_broadcast + scores + 2 + v * scores_outer_sqrt),
            axis=1,
        )

        pi_denominator = np.sum(
            win_tie_half * (1 + v * scores_outer_sqrt) /
            (scores_broadcast + scores + 2 + v * scores_outer_sqrt),
            axis=0,
        )

        scores = pi_numerator / pi_denominator
        scores = np.nan_to_num(scores, nan=tolerance)

        converged = np.allclose(scores / (scores + 1), scores_old / (scores_old + 1),
                                rtol=tolerance, atol=tolerance) or (iterations >= limit)

    return scores, v, iterations


def eigen(
        matrix: npt.NDArray[np.float64],
        tolerance: float = 1e-6,
        limit: int = 100,
) -> tuple[npt.NDArray[np.float64], int]:
    n = matrix.shape[0]

    scores = np.ones(n) / n

    converged, iterations = False, 0

    while not converged:
        iterations += 1

        scores_old = scores.copy()

        scores = matrix.T @ scores_old
        scores /= np.linalg.norm(scores) or 1

        converged = np.allclose(scores / (scores + 1), scores_old / (scores_old + 1),
                                rtol=tolerance, atol=tolerance) or (iterations >= limit)

    return scores, iterations
