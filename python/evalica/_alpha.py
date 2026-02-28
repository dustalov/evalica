from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from . import DistanceFunc, DistanceName, InsufficientRatingsError, T_distance_contra, UnknownDistanceError

if TYPE_CHECKING:
    import pandas as pd

MIN_RATERS = 2


def _as_unit_matrix(data: pd.DataFrame) -> npt.NDArray[np.object_]:
    """
    Convert input ratings to a unit-by-observer matrix.

    Args:
        data: Ratings by observer (rows) and unit (columns).

    Returns:
        A numpy matrix with units as rows and observers as columns.

    """
    valid_counts = data.notna().sum(axis=0)
    frame = data.loc[:, valid_counts > 1].T

    if frame.empty:
        raise InsufficientRatingsError()  # noqa: RSE102

    return frame.to_numpy()


def _coincidence_matrix(
    matrix_indices: npt.NDArray[np.int64],
    n_unique: int,
) -> npt.NDArray[np.float64]:
    """
    Build the coincidence matrix from coded unit values.

    Args:
        matrix_indices: The coded unit matrix with -1 for missing values.
        n_unique: The number of unique non-missing values.

    Returns:
        The coincidence matrix.

    """
    c = np.zeros((matrix_indices.shape[0], n_unique), dtype=np.float64)

    for i, unit_row in enumerate(matrix_indices):
        valid = unit_row[unit_row >= 0]
        if valid.size > 0:
            c[i] = np.bincount(valid, minlength=n_unique)

    m_u = np.sum(c, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(m_u >= MIN_RATERS, 1.0 / (m_u - 1), 0.0)

    cw = c * weights[:, np.newaxis]
    coincidence: npt.NDArray[np.float64] = cw.T @ c
    np.fill_diagonal(coincidence, coincidence.diagonal() - np.sum(cw, axis=0))

    return coincidence


def _nominal_distance(n_unique: int) -> npt.NDArray[np.float64]:
    """
    Compute nominal distance matrix.

    Args:
        n_unique: Number of unique values.

    Returns:
        Distance matrix where off-diagonal elements are 1.0 and diagonal is 0.0.

    """
    result: npt.NDArray[np.float64] = 1.0 - np.eye(n_unique)
    return result


def _ordinal_distance(coincidence: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Compute ordinal distance matrix based on cumulative frequencies.

    Args:
        coincidence: The coincidence matrix.

    Returns:
        Distance matrix based on squared differences of midpoints.

    """
    n_c = np.sum(coincidence, axis=1)
    cum_freqs = np.cumsum(n_c)
    midpoints = cum_freqs - (n_c / 2.0)
    result: npt.NDArray[np.float64] = (midpoints[:, None] - midpoints[None, :]) ** 2
    return result


def _interval_distance(unique_values: npt.NDArray[np.object_]) -> npt.NDArray[np.float64]:
    """
    Compute interval distance matrix.

    Args:
        unique_values: The sorted unique values.

    Returns:
        Distance matrix based on squared differences of values.

    """
    values = np.asarray(unique_values, dtype=np.float64)
    result: npt.NDArray[np.float64] = (values[:, None] - values[None, :]) ** 2
    return result


def _ratio_distance(unique_values: npt.NDArray[np.object_]) -> npt.NDArray[np.float64]:
    """
    Compute ratio distance matrix.

    Args:
        unique_values: The sorted unique values.

    Returns:
        Distance matrix based on squared relative differences.

    """
    values = np.asarray(unique_values, dtype=np.float64)
    sum_matrix = values[:, None] + values[None, :]
    diff_matrix = values[:, None] - values[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = (diff_matrix / sum_matrix) ** 2
    delta[np.isnan(delta)] = 0.0
    result: npt.NDArray[np.float64] = delta
    return result


def _custom_distance(
    distance_func: DistanceFunc[T_distance_contra],
    unique_values: npt.NDArray[np.object_],
) -> npt.NDArray[np.float64]:
    """
    Compute distance matrix using a custom distance function.

    Args:
        distance_func: Custom distance function.
        unique_values: The sorted unique values.

    Returns:
        Distance matrix computed using the custom function.

    """
    vectorized = np.vectorize(
        lambda left, right: float(distance_func(left, right)),
        otypes=[np.float64],
    )
    delta: npt.NDArray[np.float64] = vectorized(unique_values[:, None], unique_values[None, :])
    np.fill_diagonal(delta, 0.0)
    return delta


def _compute_delta_matrix(
    distance: Any,  # noqa: ANN401
    unique_values: npt.NDArray[np.object_],
    coincidence: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Compute the delta matrix for the requested distance metric.

    Args:
        distance: The distance name or a custom callable.
        unique_values: The sorted unique values.
        coincidence: The coincidence matrix (used only for ordinal distance).

    Returns:
        The delta matrix.

    """
    if callable(distance):
        return _custom_distance(distance, unique_values)

    if distance == "nominal":
        return _nominal_distance(len(unique_values))
    if distance == "ordinal":
        return _ordinal_distance(coincidence)
    if distance == "interval":
        return _interval_distance(unique_values)
    if distance == "ratio":
        return _ratio_distance(unique_values)

    raise UnknownDistanceError(distance)


def _compute_expected_matrix(
    coincidence: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Compute the expected disagreement matrix.

    Args:
        coincidence: The coincidence matrix.

    Returns:
        The expected disagreement matrix.

    """
    n_c = np.sum(coincidence, axis=1)
    n_total = np.sum(n_c)

    if n_total > 1:
        outer = np.outer(n_c, n_c)
        np.fill_diagonal(outer, n_c * (n_c - 1))
        result: npt.NDArray[np.float64] = outer / (n_total - 1)
        return result

    result_zeros: npt.NDArray[np.float64] = np.zeros_like(coincidence)
    return result_zeros


def _alpha_components(
    matrix_indices: npt.NDArray[np.int64],
    unique_values: npt.NDArray[np.object_],
    distance: DistanceFunc[T_distance_contra] | DistanceName,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the components required for Krippendorff's alpha.

    Args:
        matrix_indices: The coded unit matrix with -1 for missing values.
        unique_values: The unique values.
        distance: Distance metric (nominal, ordinal, interval, ratio) or a custom function.

    Returns:
        A tuple of (coincidence, expected_matrix, delta).

    """
    coincidence = _coincidence_matrix(matrix_indices, len(unique_values))
    expected_matrix = _compute_expected_matrix(coincidence)
    delta = _compute_delta_matrix(distance, unique_values, coincidence)
    return coincidence, expected_matrix, delta


def _alpha_naive(
    matrix_indices: npt.NDArray[np.int64],
    unique_values: npt.NDArray[np.object_],
    distance: DistanceFunc[T_distance_contra] | DistanceName,
) -> tuple[float, float, float]:
    """
    Compute Krippendorff's alpha (naive Python implementation).

    Args:
        matrix_indices: The coded unit matrix with -1 for missing values.
        unique_values: The unique values.
        distance: Distance metric (nominal, ordinal, interval, ratio) or a custom function.

    Returns:
        A tuple of (alpha, observed_disagreement, expected_disagreement).

    """
    coincidence, expected_matrix, delta = _alpha_components(matrix_indices, unique_values, distance)
    observed = float(np.sum(coincidence * delta))
    expected = float(np.sum(expected_matrix * delta))
    _alpha = (1.0 if observed == 0.0 else 0.0) if expected == 0.0 else float(1.0 - observed / expected)
    return (_alpha, observed, expected)


def _alpha_bootstrap_naive(
    matrix_indices: npt.NDArray[np.int64],
    unique_values: npt.NDArray[np.object_],
    distance: DistanceFunc[T_distance_contra] | DistanceName,
    n_resamples: int,
    random_state: int | None = None,
    *,
    min_resamples: int = 1000,
) -> npt.NDArray[np.float64]:
    """
    Compute confidence intervals for Krippendorff's alpha using bootstrap (naive Python).

    Args:
        matrix_indices: The coded unit matrix with -1 for missing values.
        unique_values: The unique values.
        distance: Distance metric (nominal, ordinal, interval, ratio) or a custom function.
        n_resamples: Number of bootstrap samples.
        random_state: The random seed.
        min_resamples: Minimum number of bootstrap samples.

    Returns:
        The alpha bootstrap distribution.

    """
    _, expected_matrix, delta = _alpha_components(matrix_indices, unique_values, distance)

    expected = float(np.sum(expected_matrix * delta))

    if min_resamples <= 0:
        msg = "min_resamples must be a positive integer."
        raise ValueError(msg)

    if n_resamples < min_resamples:
        msg = f"Number of resamples must be at least {min_resamples}."
        raise ValueError(msg)
    if expected <= 0.0:
        msg = "Bootstrapping is not defined when expected disagreement is zero."
        raise ValueError(msg)

    unit_counts = np.sum(matrix_indices >= 0, axis=1)
    valid_units_mask = unit_counts >= MIN_RATERS

    unit_counts = unit_counts[valid_units_mask]
    matrix_indices = matrix_indices[valid_units_mask]

    draws_per_unit = unit_counts * (unit_counts - 1) // 2
    total_draws = int(draws_per_unit.sum())

    pair_errors_list = []

    for unit_codes in matrix_indices:
        valid = unit_codes[unit_codes >= 0]
        n_valid = len(valid)

        i, j = np.triu_indices(n_valid, k=1)

        errors = 2.0 * delta[valid[i], valid[j]] / expected

        pair_errors_list.append(errors)

    pair_errors = np.concatenate(pair_errors_list)
    n_pair_errors = pair_errors.size

    rng = np.random.default_rng(random_state)

    draws = rng.integers(
        0,
        n_pair_errors,
        size=(n_resamples, total_draws),
        dtype=np.int64,
    )

    alphas = np.ones(n_resamples, dtype=np.float64)

    start = 0

    for unit_count, n_draw in zip(unit_counts, draws_per_unit):
        end = start + n_draw
        unit_draws = draws[:, start:end]

        if n_draw >= 2:  # noqa: PLR2004
            mask = unit_draws[:, 1] == unit_draws[:, 0]
            if np.any(mask):
                unit_draws[mask, 1] = rng.integers(
                    0,
                    n_pair_errors,
                    size=mask.sum(),
                    dtype=np.int64,
                )

        alphas -= pair_errors[unit_draws].sum(axis=1) / (unit_count - 1)
        start = end

    np.maximum(alphas, -1.0, out=alphas)

    result: npt.NDArray[np.float64] = alphas
    return result
