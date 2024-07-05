import numpy as np
import numpy.typing as npt


def bradley_terry(M: npt.NDArray[np.int64], tolerance: float = 1e-8) -> tuple[  # noqa: N803
    npt.NDArray[np.float64], int]:
    T = M.T + M  # noqa: N806
    active = T > 0

    w = M.sum(axis=1)

    Z = np.zeros_like(M, dtype=float)  # noqa: N806

    scores = np.ones(M.shape[0])
    scores_new = scores.copy()

    converged, iterations = False, 0

    while not converged:
        iterations += 1

        P = np.broadcast_to(scores, M.shape)  # noqa: N806

        Z[active] = T[active] / (P[active] + P.T[active])

        scores_new[:] = w
        scores_new /= Z.sum(axis=0)
        scores_new /= scores_new.sum()

        converged = bool(np.linalg.norm(scores_new - scores) < tolerance)

        scores[:] = scores_new

    return scores, iterations


def newman(W: npt.NDArray[np.int64], T: npt.NDArray[np.int64], tolerance: float = 1e-6,  # noqa: N803
           scores_init: npt.NDArray[np.float64] | None = None,
           v_init: float | None = None) -> tuple[npt.NDArray[np.float64], int]:
    rng = np.random.default_rng()

    scores = rng.random(W.shape[0]) if scores_init is None else scores_init

    v = rng.random() if v_init is None else v_init

    converged, iterations = False, 0

    while not converged:
        iterations += 1

        v_numerator = np.sum(
            T * (scores[:, np.newaxis] + scores) /
            (scores[:, np.newaxis] + scores + 2 * v * np.sqrt(scores[:, np.newaxis] * scores)),
        ) / 2

        v_denominator = np.sum(
            W * 2 * np.sqrt(scores[:, np.newaxis] * scores) /
            (scores[:, np.newaxis] + scores + 2 * v * np.sqrt(scores[:, np.newaxis] * scores)),
        )

        v = v_numerator / v_denominator

        scores_old = scores.copy()

        pi_numerator = np.sum(
            (W + T / 2) * (scores + v * np.sqrt(scores[:, np.newaxis] * scores)) /
            (scores[:, np.newaxis] + scores + 2 * v * np.sqrt(scores[:, np.newaxis] * scores)),
            axis=1,
        )

        pi_denominator = np.sum(
            (W + T / 2) * (1 + v * np.sqrt(scores[:, np.newaxis] * scores)) /
            (scores[:, np.newaxis] + scores + 2 * v * np.sqrt(scores[:, np.newaxis] * scores)),
            axis=0,
        )

        scores = pi_numerator / pi_denominator

        converged = np.allclose(scores / (scores + 1), scores_old / (scores_old + 1), rtol=tolerance, atol=tolerance)

    return scores, iterations
