from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import pytest
from hypothesis import given

import evalica
from conftest import rating_dataframes
from evalica import AlphaResult
from evalica._alpha import _coincidence_matrix, _compute_expected_matrix

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


def test_alpha_custom_distance_pyo3_error() -> None:
    data = [[1, 2], [2, 3], [3, 4]]
    df = pd.DataFrame(data).T

    with pytest.raises(evalica.SolverError, match="The 'pyo3' solver is not available"):
        evalica.alpha(df, distance=lambda _x, _y: 0.0, solver="pyo3")


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
    distance: Literal["nominal", "ordinal", "interval", "ratio"],
    solver: Literal["naive", "pyo3"],
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
    matrix_indices = np.array([
        [0, 1, -1],
        [-1, -1, -1],
        [2, 3, 4],
    ], dtype=np.int64)

    result = _coincidence_matrix(matrix_indices, n_unique=5)

    assert result.shape == (5, 5)
    assert isinstance(result, np.ndarray)


def test_compute_expected_matrix_zero_case() -> None:
    coincidence = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)

    result = _compute_expected_matrix(coincidence)

    assert result.shape == (2, 2)
    assert np.array_equal(result, np.zeros((2, 2)))
