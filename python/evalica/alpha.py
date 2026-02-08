from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import DistanceFunc, DistanceName, InsufficientRatingsError, T_distance_contra, UnknownDistanceError


def _as_unit_matrix(data: pd.DataFrame) -> npt.NDArray[np.object_]:
    """
    Convert input ratings to a unit-by-observer matrix.

    Args:
        data: Ratings by observer (rows) and unit (columns).

    Returns:
        A numpy matrix with units as rows and observers as columns.

    Raises:
        TypeError: If ``data`` is not a pandas DataFrame.
        InsufficientRatingsError: If no unit has at least two ratings.

    """
    valid_counts = data.notna().sum(axis=0)
    frame = data.loc[:, valid_counts > 1].T

    if frame.empty:
        raise InsufficientRatingsError()  # noqa: RSE102

    return frame.to_numpy()


def _factorize_values(
    matrix: npt.NDArray[np.object_],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.object_]]:
    """
    Map non-missing values to integer codes and return sorted uniques.

    Args:
        matrix: The unit matrix with object values and missing entries.

    Returns:
        A tuple of (coded matrix, unique values) where missing values are -1.

    """
    flat = matrix.ravel()
    codes, uniques = pd.factorize(flat)

    sort_idx = np.argsort(uniques)
    unique_values = uniques[sort_idx]

    remap = np.empty(len(uniques), dtype=int)
    remap[sort_idx] = np.arange(len(uniques))

    valid_mask = codes != -1
    sorted_codes = np.full_like(codes, -1)
    sorted_codes[valid_mask] = remap[codes[valid_mask]]

    return sorted_codes.reshape(matrix.shape), unique_values


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
    coincidence = np.zeros((n_unique, n_unique), dtype=np.float64)

    min_raters = 2
    for unit_values in matrix_indices:
        valid_values = unit_values[unit_values >= 0]
        m_u = len(valid_values)
        if m_u < min_raters:
            continue

        weight = 1.0 / (m_u - 1)
        counts = np.bincount(valid_values, minlength=n_unique).astype(np.float64)
        contribution = np.outer(counts, counts)
        np.fill_diagonal(contribution, np.diag(contribution) - counts)
        coincidence += weight * contribution

    return coincidence


def _nominal_distance(n_unique: int) -> npt.NDArray[np.float64]:
    """
    Compute nominal distance matrix.

    Args:
        n_unique: Number of unique values.

    Returns:
        Distance matrix where off-diagonal elements are 1.0 and diagonal is 0.0.

    """
    return 1.0 - np.eye(n_unique)


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
    return (values[:, None] - values[None, :]) ** 2


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
    return delta


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

    Raises:
        UnknownDistanceError: If an unknown distance name is provided.

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

    return np.zeros_like(coincidence)


def _alpha_naive(
    data: pd.DataFrame,
    distance: DistanceFunc[T_distance_contra] | DistanceName,
) -> tuple[float, float, float]:
    """
    Compute Krippendorff's alpha (naive Python implementation).

    Args:
        data: Ratings by observer (rows) and unit (columns).
        distance: Distance metric (nominal, ordinal, interval, ratio) or a custom function.

    Returns:
        A tuple of (alpha, observed_disagreement, expected_disagreement).

    """
    matrix = _as_unit_matrix(data)
    matrix_indices, unique_values = _factorize_values(matrix)
    coincidence = _coincidence_matrix(matrix_indices, len(unique_values))

    expected = _compute_expected_matrix(coincidence)
    delta = _compute_delta_matrix(distance, unique_values, coincidence)

    observed_disagreement = np.sum(coincidence * delta)
    expected_disagreement = np.sum(expected * delta)

    alpha_value = 0.0 if expected_disagreement == 0.0 else float(1.0 - observed_disagreement / expected_disagreement)

    return (alpha_value, float(observed_disagreement), float(expected_disagreement))
