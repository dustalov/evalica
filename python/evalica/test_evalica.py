from __future__ import annotations

import pickle
import sys
import unittest.mock

import pytest

import evalica


def test_rust_extension_warning() -> None:
    with unittest.mock.patch.dict(sys.modules, {"evalica._brzo": None}):
        if "evalica" in sys.modules:
            del sys.modules["evalica"]

        for mod in list(sys.modules.keys()):
            if mod.startswith("evalica."):
                del sys.modules[mod]

        with pytest.warns(RuntimeWarning, match="The Rust extension could not be imported"):
            import evalica  # noqa: PLC0415

        assert not evalica.PYO3_AVAILABLE
        assert evalica.SOLVER == "naive"

        with pytest.raises(evalica.SolverError, match="The 'pyo3' solver is not available"):
            evalica.counting([], [], [], solver="pyo3")


def test_version() -> None:
    assert isinstance(evalica.__version__, str)
    assert len(evalica.__version__) > 0


@pytest.mark.skipif(not evalica.PYO3_AVAILABLE, reason="Rust extension is not available")
def test_version_consistency() -> None:
    assert evalica.__version__ == evalica._brzo.__version__  # noqa: SLF001


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
