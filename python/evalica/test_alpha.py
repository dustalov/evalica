from __future__ import annotations

import sys
import unittest.mock
from functools import partial
from typing import TYPE_CHECKING

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings

import evalica
from conftest import rating_dataframes
from evalica import AlphaBootstrapResult, AlphaResult, DistanceName, SolverName
from evalica._alpha import _alpha_bootstrap_naive, _coincidence_matrix, _compute_expected_matrix

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
def test_alpha(codings: pd.DataFrame, solver: str) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    result = evalica.alpha(codings, solver=solver)  # type: ignore[arg-type]
    assert result.alpha == pytest.approx(904 / 1216)
    assert result.solver == solver


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
def test_alpha_gcl(gcl: pd.DataFrame, solver: str) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    result = evalica.alpha(gcl, solver=solver)  # type: ignore[arg-type]
    assert result.alpha == pytest.approx(0.0833333)
    assert result.solver == solver


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
def test_alpha_nominal_default(solver: str) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    data = [
        [1, 1, None, 1],
        [2, 2, 3, 2],
        [3, 3, 3, 3],
        [3, 3, 3, 3],
        [2, 2, 2, 2],
        [1, 2, 3, 4],
        [4, 4, 4, 4],
        [1, 1, 2, 1],
        [2, 2, 2, 2],
        [None, 5, 5, 5],
        [None, None, 1, 1],
    ]
    df = pd.DataFrame(data).T
    result = evalica.alpha(df, solver=solver)  # type: ignore[arg-type]
    assert result.alpha == pytest.approx(0.7434211)
    assert result.solver == solver


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
def test_alpha_interval(solver: str) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    data = [[1, 2], [2, 3], [3, 4]]
    df = pd.DataFrame(data).T

    res_nom = evalica.alpha(df, distance="nominal", solver=solver)  # type: ignore[arg-type]
    assert res_nom.alpha == pytest.approx(-0.1538462)
    assert res_nom.solver == solver

    res_int = evalica.alpha(df, distance="interval", solver=solver)  # type: ignore[arg-type]
    assert res_int.alpha == pytest.approx(0.5454545)
    assert res_int.solver == solver


def test_alpha_custom_distance() -> None:
    data = [[1, 2], [2, 3], [3, 4]]
    df = pd.DataFrame(data).T

    def custom_dist(x: float, y: float) -> float:
        return (x - y) ** 2

    res_custom = evalica.alpha(df, distance=custom_dist, solver="naive")
    assert res_custom.alpha == pytest.approx(0.5454545)
    assert res_custom.solver == "naive"


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
def test_alpha_ordinal(solver: str) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    data = [[1, 2], [2, 3]]
    df = pd.DataFrame(data).T
    res_ord = evalica.alpha(df, distance="ordinal", solver=solver)  # type: ignore[arg-type]
    res_int = evalica.alpha(df, distance="interval", solver=solver)  # type: ignore[arg-type]
    assert res_ord.alpha == res_int.alpha
    assert res_ord.solver == solver
    assert res_int.solver == solver


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
def test_alpha_ratio(solver: str) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    data = [[1, 2], [2, 4]]
    df = pd.DataFrame(data).T
    res_ratio = evalica.alpha(df, distance="ratio", solver=solver)  # type: ignore[arg-type]
    assert res_ratio.alpha == pytest.approx(0.1712707)
    assert res_ratio.solver == solver


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
def test_alpha_unknown_distance(solver: str) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    data = [[1, 2], [2, 3]]
    df = pd.DataFrame(data).T
    with pytest.raises(evalica.UnknownDistanceError, match="Unknown distance"):
        evalica.alpha(df, distance="mystery", solver=solver)  # type: ignore[arg-type]


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
def test_alpha_no_pairable_units(solver: str) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    df = pd.DataFrame([[1, None], [None, 2]])
    with pytest.raises(evalica.InsufficientRatingsError, match="No units have at least 2 ratings"):
        evalica.alpha(df, solver=solver)  # type: ignore[arg-type]


@pytest.mark.parametrize("distance", ["nominal", "ordinal", "interval", "ratio"])
def test_alpha_distances_codings(codings: pd.DataFrame, distance: str) -> None:
    if not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    result_pyo3 = evalica.alpha(codings, distance=distance, solver="pyo3")  # type: ignore[arg-type]
    result_naive = evalica.alpha(codings, distance=distance, solver="naive")  # type: ignore[arg-type]

    for result in (result_pyo3, result_naive):
        assert isinstance(result, AlphaResult)
        assert isinstance(result.alpha, float)
        assert isinstance(result.observed, float)
        assert isinstance(result.expected, float)
        assert result.solver in ("pyo3", "naive")

    assert result_pyo3.alpha == pytest.approx(result_naive.alpha)
    assert result_pyo3.observed == pytest.approx(result_naive.observed)
    assert result_pyo3.expected == pytest.approx(result_naive.expected)


@pytest.mark.parametrize("distance", ["nominal", "ordinal", "interval", "ratio"])
def test_alpha_distances_gcl(gcl: pd.DataFrame, distance: str) -> None:
    if not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    result_pyo3 = evalica.alpha(gcl, distance=distance, solver="pyo3")  # type: ignore[arg-type]
    result_naive = evalica.alpha(gcl, distance=distance, solver="naive")  # type: ignore[arg-type]

    for result in (result_pyo3, result_naive):
        assert isinstance(result, AlphaResult)
        assert isinstance(result.alpha, float)
        assert isinstance(result.observed, float)
        assert isinstance(result.expected, float)
        assert result.solver in ("pyo3", "naive")

    assert result_pyo3.alpha == pytest.approx(result_naive.alpha)
    assert result_pyo3.observed == pytest.approx(result_naive.observed)
    assert result_pyo3.expected == pytest.approx(result_naive.expected)


@pytest.mark.parametrize(
    ("distance", "solver"),
    [
        ("nominal", "naive"),
        ("nominal", "pyo3"),
        ("ordinal", "naive"),
        ("ordinal", "pyo3"),
        ("interval", "naive"),
        ("interval", "pyo3"),
        ("ratio", "naive"),
        ("ratio", "pyo3"),
    ],
)
@given(data=rating_dataframes())
def test_alpha_properties(data: pd.DataFrame, distance: str, solver: str) -> None:
    if not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    result = evalica.alpha(data, distance=distance, solver=solver)  # type: ignore[arg-type]

    assert isinstance(result, AlphaResult)
    assert isinstance(result.alpha, float)
    assert np.isfinite(result.alpha)
    assert result.observed >= 0.0
    assert result.expected >= 0.0
    assert result.solver == solver


@pytest.mark.skipif(not evalica.PYO3_AVAILABLE, reason="Rust extension is not available")
@pytest.mark.parametrize("distance", ["nominal", "ordinal", "interval", "ratio"])
@given(data=rating_dataframes())
def test_alpha_solvers_hypothesis(data: pd.DataFrame, distance: str) -> None:
    result_pyo3 = evalica.alpha(data, distance=distance, solver="pyo3")  # type: ignore[arg-type]
    result_naive = evalica.alpha(data, distance=distance, solver="naive")  # type: ignore[arg-type]

    assert result_pyo3.alpha == pytest.approx(result_naive.alpha)
    assert result_pyo3.observed == pytest.approx(result_naive.observed)
    assert result_pyo3.expected == pytest.approx(result_naive.expected)


@pytest.mark.skipif(not evalica.PYO3_AVAILABLE, reason="Rust extension is not available")
@settings(max_examples=12, deadline=None)
@given(
    data=rating_dataframes(),
    distance=st.sampled_from(["nominal", "ordinal", "interval", "ratio"]),
)
def test_alpha_bootstrap_solvers_hypothesis(
    data: pd.DataFrame,
    distance: DistanceName,
) -> None:
    alpha_pyo3 = evalica.alpha(data, distance=distance, solver="pyo3")
    alpha_naive = evalica.alpha(data, distance=distance, solver="naive")
    assume(alpha_pyo3.expected > 0.0 and alpha_naive.expected > 0.0)

    n_resamples = 1000
    confidence_level = 0.95
    random_state = 42

    result_pyo3 = evalica.alpha_bootstrap(
        data,
        distance=distance,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_state=random_state,
        solver="pyo3",
    )
    result_naive = evalica.alpha_bootstrap(
        data,
        distance=distance,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_state=random_state,
        solver="naive",
    )

    for result in (result_pyo3, result_naive):
        assert isinstance(result, AlphaBootstrapResult)
        assert isinstance(result, AlphaResult)
        assert result.n_resamples == n_resamples
        assert result.confidence_level == confidence_level
        assert len(result.distribution) == n_resamples
        assert np.isfinite(result.distribution).all()

    assert result_pyo3.alpha == pytest.approx(result_naive.alpha)
    assert result_pyo3.observed == pytest.approx(result_naive.observed)
    assert result_pyo3.expected == pytest.approx(result_naive.expected)
    assert np.all(result_pyo3.distribution >= -1.0)
    assert np.all(result_naive.distribution >= -1.0)


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
@given(data=rating_dataframes())
def test_alpha_perfect_agreement_bounds(data: pd.DataFrame, solver: str) -> None:
    if not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    result = evalica.alpha(data, distance="nominal", solver=solver)  # type: ignore[arg-type]

    assert result.alpha <= 1.0


@pytest.mark.parametrize(
    ("distance", "solver"),
    [
        ("nominal", "naive"),
        ("nominal", "pyo3"),
        ("ordinal", "naive"),
        ("ordinal", "pyo3"),
        ("interval", "naive"),
        ("interval", "pyo3"),
        ("ratio", "naive"),
        ("ratio", "pyo3"),
    ],
)
def test_alpha_performance(
    codings: pd.DataFrame,
    distance: DistanceName,
    solver: SolverName,
    benchmark: BenchmarkFixture,
) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    func = partial(evalica.alpha, codings, distance=distance, solver=solver)

    benchmark(func)


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
def test_alpha_non_numeric(solver: str) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    df = pd.DataFrame([["A", "B"], ["B", "A"], ["A", "A"]]).T
    result = evalica.alpha(df, distance="nominal", solver=solver)  # type: ignore[arg-type]
    assert isinstance(result.alpha, float)


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
def test_alpha_zero_expected(solver: str) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    df = pd.DataFrame([["A", "A"], ["A", "A"]]).T
    result = evalica.alpha(df, distance="nominal", solver=solver)  # type: ignore[arg-type]
    assert result.expected == 0.0
    assert result.alpha == 1.0


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
def test_alpha_single_rater_units(solver: str) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    df = pd.DataFrame(
        {
            "rater1": ["A", "B", np.nan],
            "rater2": ["A", "B", np.nan],
            "rater3": [np.nan, np.nan, "C"],
        },
    ).T
    result = evalica.alpha(df, distance="nominal", solver=solver)  # type: ignore[arg-type]
    assert isinstance(result, AlphaResult)
    assert result.alpha == 1.0


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
def test_alpha_n_total_edge_case(solver: str) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    df = pd.DataFrame([[1.0], [1.0]])
    result = evalica.alpha(df, distance="nominal", solver=solver)  # type: ignore[arg-type]
    assert isinstance(result, AlphaResult)
    assert result.expected == 0.0
    assert result.alpha == 1.0


def test_coincidence_matrix_skip_insufficient_raters() -> None:
    matrix_indices = np.array(
        [
            [0, 1, -1],
            [-1, -1, -1],
            [2, 3, 4],
        ],
        dtype=np.int64,
    )

    result = _coincidence_matrix(matrix_indices, n_unique=5)

    assert result.shape == (5, 5)
    assert isinstance(result, np.ndarray)


def test_compute_expected_matrix_zero_case() -> None:
    coincidence = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)

    result = _compute_expected_matrix(coincidence)

    assert result.shape == (2, 2)
    assert np.array_equal(result, np.zeros((2, 2)))


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
def test_alpha_custom_distance_function(solver: str) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    data = [
        [1, 1, None, 1],
        [2, 2, 3, 2],
        [3, 3, 3, 3],
        [3, 3, 3, 3],
        [2, 2, 2, 2],
    ]
    df = pd.DataFrame(data)

    def custom_squared_diff(left: float, right: float) -> float:
        return float((left - right) ** 2)

    result = evalica.alpha(df, distance=custom_squared_diff, solver=solver)  # type: ignore[arg-type]

    assert isinstance(result, AlphaResult)
    assert isinstance(result.alpha, float)
    assert np.isfinite(result.alpha)
    assert result.solver == solver


@pytest.mark.skipif(not evalica.PYO3_AVAILABLE, reason="Rust extension is not available")
def test_alpha_custom_distance_solvers_match() -> None:
    data = [
        [1, 2, 3, 2],
        [2, 3, 4, 3],
        [3, 4, 5, 4],
        [1, 1, 1, 1],
    ]
    df = pd.DataFrame(data)

    def custom_abs_diff(left: float, right: float) -> float:
        return abs(float(left - right))

    result_pyo3 = evalica.alpha(df, distance=custom_abs_diff, solver="pyo3")
    result_naive = evalica.alpha(df, distance=custom_abs_diff, solver="naive")

    assert result_pyo3.alpha == pytest.approx(result_naive.alpha)
    assert result_pyo3.observed == pytest.approx(result_naive.observed)
    assert result_pyo3.expected == pytest.approx(result_naive.expected)


def test_alpha_solver_error() -> None:
    with unittest.mock.patch.dict(sys.modules, {"evalica._brzo": None}):
        sys.modules.pop("evalica", None)

        with pytest.warns():
            import evalica  # noqa: PLC0415

        assert not evalica.PYO3_AVAILABLE

        data = pd.DataFrame([[1, 2], [2, 3]])

        with pytest.raises(evalica.SolverError, match="The 'pyo3' solver is not available"):
            evalica.alpha(data, solver="pyo3")


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
def test_alpha_bootstrap_nominal_reference(solver: str) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    data = [
        [1, 1, None, 1],
        [2, 2, 3, 2],
        [3, 3, 3, 3],
        [3, 3, 3, 3],
        [2, 2, 2, 2],
        [1, 2, 3, 4],
        [4, 4, 4, 4],
        [1, 1, 2, 1],
        [2, 2, 2, 2],
        [None, 5, 5, 5],
        [None, None, 1, 1],
    ]
    df = pd.DataFrame(data).T
    result = evalica.alpha_bootstrap(df, distance="nominal", n_resamples=5000, random_state=12345, solver=solver)  # type: ignore[arg-type]

    assert isinstance(result, AlphaBootstrapResult)
    assert isinstance(result, AlphaResult)
    assert result.alpha == pytest.approx(0.7434211)
    assert 0.55 <= result.low <= 0.68
    assert 0.82 <= result.high <= 0.91
    assert result.n_resamples == 5000
    assert result.confidence_level == 0.95
    assert len(result.distribution) == 5000
    assert result.solver == solver


@pytest.mark.skipif(not evalica.PYO3_AVAILABLE, reason="Rust extension is not available")
@pytest.mark.parametrize("distance", ["nominal", "ordinal", "interval", "ratio"])
def test_alpha_bootstrap_solvers_match(
    codings: pd.DataFrame,
    distance: DistanceName,
) -> None:
    n_resamples = 1000
    confidence_level = 0.95
    random_state = 42

    result_pyo3 = evalica.alpha_bootstrap(
        codings,
        distance=distance,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_state=random_state,
        solver="pyo3",
    )
    result_naive = evalica.alpha_bootstrap(
        codings,
        distance=distance,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_state=random_state,
        solver="naive",
    )

    for result in (result_pyo3, result_naive):
        assert isinstance(result, AlphaBootstrapResult)
        assert isinstance(result, AlphaResult)
        assert result.n_resamples == n_resamples
        assert result.confidence_level == confidence_level
        assert len(result.distribution) == n_resamples

    assert result_pyo3.alpha == pytest.approx(result_naive.alpha)
    assert result_pyo3.observed == pytest.approx(result_naive.observed)
    assert result_pyo3.expected == pytest.approx(result_naive.expected)
    assert result_pyo3.low == pytest.approx(result_naive.low, abs=0.02)
    assert result_pyo3.high == pytest.approx(result_naive.high, abs=0.02)

    pyo3_quantiles = np.quantile(result_pyo3.distribution, [0.025, 0.5, 0.975])
    naive_quantiles = np.quantile(result_naive.distribution, [0.025, 0.5, 0.975])
    assert pyo3_quantiles == pytest.approx(naive_quantiles, abs=0.02)


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
def test_alpha_bootstrap_no_truncation(solver: SolverName) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    df = pd.DataFrame([[1, 1, 2], [2, 2, 2], [3, 3, 3], [1, 2, 1]]).T
    result = evalica.alpha_bootstrap(
        df,
        n_resamples=2300,
        random_state=7,
        solver=solver,
    )
    assert result.n_resamples == 2300
    assert len(result.distribution) == 2300


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
def test_alpha_bootstrap_zero_expected_error(solver: SolverName) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    df = pd.DataFrame([["A", "A"], ["A", "A"]]).T
    with pytest.raises(ValueError, match="expected disagreement is zero"):
        evalica.alpha_bootstrap(df, n_resamples=1000, solver=solver)


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
def test_alpha_bootstrap_negative_random_state_error(solver: SolverName) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    df = pd.DataFrame([[1, 2], [2, 1], [1, 1]]).T
    with pytest.raises(ValueError, match="non-negative integer"):
        evalica.alpha_bootstrap(df, n_resamples=1000, random_state=-1, solver=solver)


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
def test_alpha_bootstrap_invalid_n_resamples_error(solver: SolverName) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    df = pd.DataFrame([[1, 2], [2, 1], [1, 1]]).T
    with pytest.raises(ValueError, match="n_resamples must be a positive integer"):
        evalica.alpha_bootstrap(df, n_resamples=0, solver=solver)


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
def test_alpha_bootstrap_invalid_confidence_level_error(solver: SolverName) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    df = pd.DataFrame([[1, 2], [2, 1], [1, 1]]).T
    with pytest.raises(ValueError, match="confidence_level must be in"):
        evalica.alpha_bootstrap(df, n_resamples=1000, confidence_level=1.5, solver=solver)


@pytest.mark.skipif(not evalica.PYO3_AVAILABLE, reason="Rust extension is not available")
def test_alpha_bootstrap_pyo3_custom_distance() -> None:
    data = [[1, 2], [2, 3], [3, 4]]
    df = pd.DataFrame(data).T

    def custom_dist(x: float, y: float) -> float:
        return (x - y) ** 2

    res_custom = evalica.alpha_bootstrap(df, distance=custom_dist, solver="pyo3", n_resamples=1000)
    assert isinstance(res_custom, AlphaBootstrapResult)
    assert res_custom.alpha == pytest.approx(0.5454545)
    assert res_custom.solver == "pyo3"


def test_alpha_bootstrap_solver_error() -> None:
    with unittest.mock.patch.dict(sys.modules, {"evalica._brzo": None}):
        sys.modules.pop("evalica", None)

        with pytest.warns():
            import evalica  # noqa: PLC0415

        assert not evalica.PYO3_AVAILABLE

        data = pd.DataFrame([[1, 2], [2, 3]])

        with pytest.raises(evalica.SolverError, match="The 'pyo3' solver is not available"):
            evalica.alpha_bootstrap(data, solver="pyo3")


def test_alpha_bootstrap_naive_internal_checks() -> None:
    matrix_indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
    unique_values = np.array([1, 2], dtype=np.object_)

    with pytest.raises(ValueError, match="n_resamples must be a positive integer"):
        _alpha_bootstrap_naive(matrix_indices, unique_values, "nominal", 0)

    dist = _alpha_bootstrap_naive(matrix_indices, unique_values, "nominal", 1000)
    assert len(dist) == 1000
