import numpy as np
import numpy.typing as npt


def bradley_terry(
        matrix: npt.NDArray[np.float64],
        tolerance: float = 1e-8,
        limit: int = 100,
) -> tuple[npt.NDArray[np.float64], int]:
    T = matrix.T + matrix  # noqa: N806
    active = T > 0

    w = matrix.sum(axis=1)

    Z = np.zeros_like(matrix, dtype=float)  # noqa: N806

    scores = np.ones(matrix.shape[0])
    scores_new = scores.copy()

    converged, iterations = False, 0

    while not converged and iterations < limit:
        iterations += 1

        P = np.broadcast_to(scores, matrix.shape)  # noqa: N806

        Z[active] = T[active] / (P[active] + P.T[active])

        scores_new[:] = w
        scores_new /= Z.sum(axis=0)
        scores_new /= scores_new.sum()

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

        v_numerator = np.sum(
            tie_matrix * (scores[:, np.newaxis] + scores) /
            (scores[:, np.newaxis] + scores + 2 * v * np.sqrt(scores[:, np.newaxis] * scores)),
        ) / 2

        v_denominator = np.sum(
            win_matrix * 2 * np.sqrt(scores[:, np.newaxis] * scores) /
            (scores[:, np.newaxis] + scores + 2 * v * np.sqrt(scores[:, np.newaxis] * scores)),
        )

        v = v_numerator / v_denominator
        v = np.nan_to_num(v, nan=tolerance)

        scores_old = scores.copy()

        pi_numerator = np.sum(
            win_tie_half * (scores + v * np.sqrt(scores[:, np.newaxis] * scores)) /
            (scores[:, np.newaxis] + scores + 2 + v * np.sqrt(scores[:, np.newaxis] * scores)),
            axis=1,
        )

        pi_denominator = np.sum(
            win_tie_half * (1 + v * np.sqrt(scores[:, np.newaxis] * scores)) /
            (scores[:, np.newaxis] + scores + 2 + v * np.sqrt(scores[:, np.newaxis] * scores)),
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
