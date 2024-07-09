from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import evalica
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from hypothesis import strategies as st

if TYPE_CHECKING:
    from collections.abc import Callable

    from _pytest.fixtures import TopRequest


class Example(NamedTuple):
    """A tuple holding example data."""

    xs: list[str] | pd.Series[str]
    ys: list[str] | pd.Series[str]
    ws: list[evalica.Winner] | pd.Series[evalica.Winner]  # type: ignore[type-var]


@st.composite
def elements(draw: Callable[[st.SearchStrategy[Any]], Any]) -> Example:  # type: ignore[type-var]
    length = draw(st.integers(0, 5))

    xys = st.lists(st.text(max_size=length), min_size=length, max_size=length)
    ws = st.lists(st.sampled_from(evalica.WINNERS), min_size=length, max_size=length)

    return Example(
        xs=draw(xys),
        ys=draw(xys),
        ws=draw(ws),
    )


@pytest.fixture()
def simple() -> npt.NDArray[np.int64]:
    return np.array([
        [0, 1, 2, 0, 1],
        [2, 0, 2, 1, 0],
        [1, 2, 0, 0, 1],
        [1, 2, 1, 0, 2],
        [2, 0, 1, 3, 0],
    ], dtype=np.int64)


@pytest.fixture()
def simple_tied(simple: npt.NDArray[np.int64]) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    tie_matrix = np.minimum(simple, simple.T).astype(np.int64)
    win_matrix = simple - tie_matrix

    return win_matrix, tie_matrix


def matrix_to_elements(
        matrix: npt.NDArray[np.int64],
        winner_func: Callable[[int, int], evalica.Winner],
) -> tuple[list[str], list[str], list[evalica.Winner]]:
    xs, ys, ws = [], [], []

    for x, y in zip(*np.nonzero(matrix), strict=False):
        winner = winner_func(x, y)

        for _ in range(matrix[x, y]):
            xs.append(str(x))
            ys.append(str(y))
            ws.append(winner)

    return xs, ys, ws


@pytest.fixture()
def simple_elements(simple: npt.NDArray[np.int64]) -> Example:
    xs, ys, ws = matrix_to_elements(simple, lambda x, y: evalica.Winner.X if x > y else evalica.Winner.Y)
    return Example(xs=xs, ys=ys, ws=ws)


@pytest.fixture()
def simple_tied_elements(simple_tied: tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]) -> Example:
    win_matrix, tie_matrix = simple_tied

    win_xs, win_ys, win_ws = matrix_to_elements(
        win_matrix,
        lambda x, y: evalica.Winner.X if x > y else evalica.Winner.Y,
    )

    tie_xs, tie_ys, tie_ws = matrix_to_elements(
        np.triu(tie_matrix),
        lambda _x, _y: evalica.Winner.Draw,
    )

    return Example(xs=win_xs + tie_xs, ys=win_ys + tie_ys, ws=win_ws + tie_ws)


@pytest.fixture()
def food() -> Example:
    df_food = pd.read_csv(Path(__file__).resolve().parent / "food.csv", dtype=str)

    xs = df_food["left"]
    ys = df_food["right"]
    ws = df_food["winner"].map({
        "left": evalica.Winner.X,
        "right": evalica.Winner.Y,
        "tie": evalica.Winner.Draw,
    })

    return Example(xs=xs, ys=ys, ws=ws)


@pytest.fixture()
def food_golden() -> pd.DataFrame:
    df_golden = pd.read_csv(Path(__file__).resolve().parent / "food-golden.csv", dtype=str)

    df_golden["score"] = df_golden["score"].astype(float)

    return df_golden


@pytest.fixture()
def llmfao() -> Example:
    df_llmfao = pd.read_csv("https://github.com/dustalov/llmfao/raw/master/crowd-comparisons.csv", dtype=str)

    xs = df_llmfao["left"]
    ys = df_llmfao["right"]
    ws = df_llmfao["winner"].map({
        "left": evalica.Winner.X,
        "right": evalica.Winner.Y,
        "tie": evalica.Winner.Draw,
    })

    return Example(xs=xs, ys=ys, ws=ws)


@pytest.fixture()
def llmfao_golden() -> pd.DataFrame:
    df_golden = pd.read_csv(Path(__file__).resolve().parent / "llmfao-golden.csv", dtype=str)

    df_golden["score"] = df_golden["score"].astype(float)

    return df_golden


DATASETS = frozenset(("food", "llmfao"))


@pytest.fixture()
def example(request: TopRequest, dataset: str) -> Example:
    assert dataset in DATASETS, f"unknown dataset: {dataset}"

    return cast(Example, request.getfixturevalue(dataset))


@pytest.fixture()
def example_golden(request: TopRequest, dataset: str, algorithm: str) -> pd.Series[str]:
    assert dataset in DATASETS, f"unknown dataset: {dataset}"

    df_golden = cast(pd.DataFrame, request.getfixturevalue(f"{dataset}_golden"))

    df_slice = df_golden[df_golden["algorithm"] == algorithm][["item", "score"]].set_index("item")

    series = cast("pd.Series[str]", df_slice.squeeze())
    series.index.name = None
    series.name = algorithm

    return series
