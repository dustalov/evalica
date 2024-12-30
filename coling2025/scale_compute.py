#!/usr/bin/env python3

from functools import partial
from pathlib import Path
from timeit import timeit
from typing import TYPE_CHECKING, Any, cast

import evalica
import pandas as pd
from scale_data import REPETITIONS, SCALE
from tqdm.auto import trange

if TYPE_CHECKING:
    from collections.abc import Callable

ALGORITHMS = [
    evalica.counting,
    evalica.average_win_rate,
    evalica.bradley_terry,
    evalica.elo,
    evalica.eigen,
    evalica.pagerank,
    evalica.newman,
]


def main() -> None:
    results = []

    for scale in trange(SCALE, desc="scale"):
        for i in range(REPETITIONS):
            with (Path("scale") / f"scale_{scale}_{i}.parquet").open("rb") as f:
                df_sample = pd.read_parquet(f)

                df_sample["winner"] = df_sample["winner"].map({
                    "model_a": evalica.Winner.X,
                    "model_b": evalica.Winner.Y,
                    "tie": evalica.Winner.Draw,
                    "tie (bothbad)": evalica.Winner.Draw,
                })

                _, _, index = evalica.indexing(df_sample["model_a"], df_sample["model_b"])

                for algorithm in ALGORITHMS:
                    stmt = partial(
                        cast("Callable[..., Any]", algorithm),
                        xs=df_sample["model_a"],
                        ys=df_sample["model_b"],
                        winners=df_sample["winner"],
                        index=index,
                        solver="pyo3",
                    )

                    time = timeit(stmt, number=1)

                    results.append((algorithm.__name__, scale, i, len(df_sample), len(index), time))

    df_results = pd.DataFrame(results, columns=["algorithm", "scale", "i", "rows", "models", "time"])
    df_results.to_csv("scale.csv", index=False)


if __name__ == "__main__":
    main()
