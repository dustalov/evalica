from __future__ import annotations

import os
import pickle
import sys
import unittest.mock
import warnings
from functools import partial

import numpy as np
import pandas as pd
import pytest

import evalica


def test_rust_extension_warning() -> None:
    with unittest.mock.patch.dict(sys.modules, {"evalica._brzo": None}):
        sys.modules.pop("evalica", None)

        with pytest.warns() as record:
            import evalica  # noqa: PLC0415

        assert any(isinstance(w.message, evalica.RustExtensionWarning) for w in record)

        assert not evalica.PYO3_AVAILABLE
        assert evalica.SOLVER == "naive"

        with pytest.raises(evalica.SolverError):
            evalica.counting([], [], [], solver="pyo3")


def test_version() -> None:
    assert isinstance(evalica.__version__, str)
    assert len(evalica.__version__) > 0


def test_has_blas() -> None:
    assert isinstance(evalica.HAS_BLAS, bool)


@pytest.mark.parametrize(
    "algorithm",
    ["counting", "average_win_rate", "bradley_terry", "newman", "elo", "eigen", "pagerank"],
)
def test_solver_errors_all_functions(algorithm: str) -> None:
    with unittest.mock.patch.dict(sys.modules, {"evalica._brzo": None}):
        sys.modules.pop("evalica", None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            import evalica  # noqa: PLC0415

        assert not evalica.PYO3_AVAILABLE

        func = partial(
            getattr(evalica, algorithm),
            ["a", "b", "c"],
            ["b", "c", "a"],
            [evalica.Winner.X, evalica.Winner.Y, evalica.Winner.Draw],
            solver="pyo3",
        )

        with pytest.raises(evalica.SolverError):
            func()


@pytest.mark.skipif(not evalica.PYO3_AVAILABLE, reason="Rust extension is not available")
def test_brzo_has_version() -> None:
    assert hasattr(evalica._brzo, "__version__")  # noqa: SLF001
    assert isinstance(evalica._brzo.__version__, str)  # noqa: SLF001


def test_exports() -> None:
    for attr in evalica.__all__:
        assert hasattr(evalica, attr), f"missing attribute: {attr}"


def test_winner_hashable() -> None:
    assert len(evalica.WINNERS) == len(set(evalica.WINNERS))


def test_winner_pickle() -> None:
    for w in evalica.WINNERS:
        dumped = pickle.dumps(w)
        loaded = pickle.loads(dumped)  # noqa: S301
        assert w == loaded


def test_envvar_disable_pyo3() -> None:
    original_evalica = sys.modules.get("evalica")
    original_brzo = sys.modules.get("evalica._brzo")

    try:
        with unittest.mock.patch.dict(os.environ, {"EVALICA_NIJE_BRZO": "1"}):
            sys.modules.pop("evalica", None)
            sys.modules.pop("evalica._brzo", None)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                import evalica  # noqa: PLC0415

            assert not evalica.PYO3_AVAILABLE
            assert evalica.SOLVER == "naive"
    finally:
        if original_evalica is not None:
            sys.modules["evalica"] = original_evalica
        if original_brzo is not None:
            sys.modules["evalica._brzo"] = original_brzo


@pytest.mark.skipif(not evalica.PYO3_AVAILABLE, reason="Rust extension is not available")
def test_matrices_solver_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evalica, "PYO3_AVAILABLE", False)
    xs_indexed, ys_indexed, index = [0], [1], pd.Index(["a", "b"])
    with pytest.raises(evalica.SolverError):
        evalica.matrices(xs_indexed, ys_indexed, [evalica.Winner.X], index, solver="pyo3")


@pytest.mark.skipif(not evalica.PYO3_AVAILABLE, reason="Rust extension is not available")
def test_pairwise_scores_solver_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evalica, "PYO3_AVAILABLE", False)
    with pytest.raises(evalica.SolverError):
        evalica.pairwise_scores(np.array([1.0, 2.0, 3.0]), solver="pyo3")


def test_version_without_brzo() -> None:
    with unittest.mock.patch.dict(sys.modules, {"evalica._brzo": None}):
        sys.modules.pop("evalica", None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            import evalica  # noqa: PLC0415

        assert not evalica.PYO3_AVAILABLE
        assert isinstance(evalica.__version__, str)
        assert len(evalica.__version__) > 0
