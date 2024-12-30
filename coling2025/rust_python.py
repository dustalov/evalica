#!/usr/bin/env python3

from functools import partial
from timeit import repeat

import evalica
import pandas as pd
from tqdm.auto import tqdm

ALGORITHMS = [
    evalica.counting,
    evalica.average_win_rate,
    evalica.bradley_terry,
    evalica.elo,
    evalica.eigen,
    evalica.pagerank,
    evalica.newman,
]

REPETITIONS = 10

def main() -> None:
    df_llmfao = pd.read_csv("llmfao.csv", dtype=str)
    df_llmfao = df_llmfao[["left", "right", "winner"]]
    df_llmfao["winner"] = df_llmfao["winner"].map({
        "left": evalica.Winner.X,
        "right": evalica.Winner.Y,
        "tie": evalica.Winner.Draw,
    })

    _, _, index = evalica.indexing(df_llmfao["left"], df_llmfao["right"])

    results = []

    for algorithm in tqdm(ALGORITHMS):
        for solver in ("pyo3", "naive"):
            stmt = partial(
                algorithm,
                xs=df_llmfao["left"],
                ys=df_llmfao["right"],
                winners=df_llmfao["winner"],
                index=index,
                solver=solver,
            )

            time = repeat(stmt, repeat=REPETITIONS, number=1)

            results.append((algorithm.__name__, solver, time))

    df_results = pd.DataFrame(results, columns=["algorithm", "solver", "time"])
    df_results = df_results.explode("time")
    df_results = df_results.reset_index(drop=True)
    df_results.to_csv("rust_python.csv", index=False)


if __name__ == "__main__":
    main()
