from pathlib import Path

import pandas as pd
import pytest

import evalica
from evalica.alpha import AlphaResult


@pytest.fixture
def codings() -> pd.DataFrame:
    return pd.read_csv(Path(__file__).resolve().parent.parent.parent / "codings.csv", header=None, dtype=str)


@pytest.mark.parametrize("solver", ["naive", "pyo3"])
def test_alpha(codings: pd.DataFrame, solver: str) -> None:
    if solver == "pyo3" and not evalica.PYO3_AVAILABLE:
        pytest.skip("Rust extension is not available")

    result = evalica.alpha(codings, solver=solver)  # type: ignore[arg-type]
    assert result.alpha == pytest.approx(0.7434211)
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

    def custom_dist(x: object, y: object) -> float:
        return (float(x) - float(y))**2  # type: ignore[arg-type]

    res_custom = evalica.alpha(df, distance=custom_dist, solver="naive")
    assert res_custom.alpha == pytest.approx(0.5454545)
    assert res_custom.solver == "naive"


def test_alpha_custom_distance_pyo3_error() -> None:
    """Test that pyo3 solver raises SolverError with custom distance functions."""
    data = [[1, 2], [2, 3], [3, 4]]
    df = pd.DataFrame(data).T

    def custom_dist(x: object, y: object) -> float:
        return (float(x) - float(y))**2  # type: ignore[arg-type]

    with pytest.raises(evalica.SolverError, match="The 'pyo3' solver is not available"):
        evalica.alpha(df, distance=custom_dist, solver="pyo3")


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
def test_alpha_solvers(codings: pd.DataFrame, distance: str) -> None:
    """Compare pyo3 and naive solvers for alpha computation."""
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
