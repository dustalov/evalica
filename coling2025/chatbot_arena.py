#!/usr/bin/env python3

# ruff: noqa: E501, EM101, F401, N803

from __future__ import annotations

import math
from collections import defaultdict  # noqa: TC003
from functools import partial
from timeit import repeat

import evalica
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

REPETITIONS = 10


def chatbot_arena_elo(
        battles: pd.DataFrame,
        K: float = 4,
        SCALE: float = 400,
        BASE: float = 10,
        INIT_RATING: float = 1000,
) -> defaultdict[str, float]:
    raise NotImplementedError(
        "Please copy the code from the official Chatbot Arena notebook and paste it here: "
        "https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH "
        "(compute_online_elo function)",
    )


def arena_hard_bradley_terry(
        df: pd.DataFrame,
        SCALE: float = 400,
        BASE: float = 10,
        INIT_RATING: float = 1000,
) -> pd.Series[str]:
    raise NotImplementedError(
        "Please copy the code from the official Arena-Hard repository and paste it here: "
        "https://github.com/lmarena/arena-hard-auto/blob/2971e34d066f986c09bc5a463fa286fa93bcca3c/utils_math.py#L38-L69",
    )


def main() -> None:
    df_arena = pd.read_json("clean_battle_20240814_public.json")
    df_arena = df_arena[df_arena["anony"]]
    df_arena = df_arena[df_arena["dedup_tag"].apply(lambda x: x.get("sampled", False))]
    df_arena["evalica"] = df_arena["winner"].map({
        "model_a": evalica.Winner.X,
        "model_b": evalica.Winner.Y,
        "tie": evalica.Winner.Draw,
        "tie (bothbad)": evalica.Winner.Draw,
    })
    df_arena = df_arena[~df_arena["evalica"].isna()]

    results = []

    with tqdm(total=4) as pbar:
        arena_elo_time = repeat(
            partial(chatbot_arena_elo, df_arena),
            repeat=REPETITIONS, number=1,
        )
        results.append(("elo", "arena", arena_elo_time))
        pbar.update()

        hard_arena_bt_time = repeat(
            partial(arena_hard_bradley_terry, df_arena),
            repeat=REPETITIONS, number=1,
        )
        results.append(("bradley_terry", "arena", hard_arena_bt_time))
        pbar.update()

        evalica_elo_time = repeat(
            partial(evalica.elo, df_arena["model_a"], df_arena["model_b"], df_arena["evalica"]),
            repeat=REPETITIONS, number=1,
        )
        results.append(("elo", "evalica", evalica_elo_time))
        pbar.update()

        evalica_bt_time = repeat(
            partial(evalica.bradley_terry, df_arena["model_a"], df_arena["model_b"], df_arena["evalica"]),
            repeat=REPETITIONS, number=1,
        )
        results.append(("bradley_terry", "evalica", evalica_bt_time))
        pbar.update()

    df_results = pd.DataFrame(results, columns=["algorithm", "solver", "time"])
    df_results = df_results.explode("time")
    df_results = df_results.reset_index(drop=True)
    df_results.to_csv("chatbot_arena.csv", index=False)


if __name__ == "__main__":
    main()
