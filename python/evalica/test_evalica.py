from __future__ import annotations

import pickle
import sys
import unittest.mock

import pytest

import evalica


def test_rust_extension_warning() -> None:
    with unittest.mock.patch.dict(sys.modules, {"evalica._brzo": None}):
        sys.modules.pop("evalica", None)

        with pytest.warns() as record:
            import evalica  # noqa: PLC0415

        assert any(
            isinstance(w.message, evalica.RustExtensionWarning)
            for w in record
        )

        assert not evalica.PYO3_AVAILABLE
        assert evalica.SOLVER == "naive"

        with pytest.raises(evalica.SolverError):
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
