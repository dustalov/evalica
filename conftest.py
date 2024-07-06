from collections.abc import Callable
from typing import Any, NamedTuple

import evalica
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy


@pytest.fixture()
def simple() -> npt.NDArray[np.float64]:
    return np.array([
        [0, 1, 2, 0, 1],
        [2, 0, 2, 1, 0],
        [1, 2, 0, 0, 1],
        [1, 2, 1, 0, 2],
        [2, 0, 1, 3, 0],
    ], dtype=np.float64)


@pytest.fixture()
def simple_win_tie(simple: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    T = np.minimum(simple, simple.T).astype(np.float64)  # noqa: N806
    W = simple - T  # noqa: N806

    return W, T


@pytest.fixture()
def food() -> tuple[list[str], list[str], list[evalica.Winner]]:
    df_food = pd.read_csv("food.csv", dtype=str)

    xs = df_food["left"]
    ys = df_food["right"]
    ws = df_food["winner"].map({
        "left": evalica.Winner.X,
        "right": evalica.Winner.Y,
        "tie": evalica.Winner.Draw,
    })

    return xs.tolist(), ys.tolist(), ws.tolist()


class Example(NamedTuple):
    """A tuple holding example data."""

    xs: list[int]
    ys: npt.NDArray[np.int64]
    ws: list[evalica.Winner]


@st.composite
def xs_ys_ws(draw: Callable[[SearchStrategy[Any]], Any]) -> Example:
    length = draw(st.integers(0, 5))
    elements = st.lists(st.integers(0, length), min_size=length, max_size=length)
    winners = st.lists(st.sampled_from(evalica.WINNERS), min_size=length, max_size=length)

    return Example(
        xs=draw(elements),
        ys=np.array(draw(elements)),
        ws=draw(winners),
    )
