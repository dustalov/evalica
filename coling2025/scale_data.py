#!/usr/bin/env python3

from pathlib import Path

import evalica
import pandas as pd
from tqdm.auto import trange

SCALE = 7

REPETITIONS = 10


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

    for scale in trange(SCALE):
        for i in range(REPETITIONS):
            with (Path("scale") / f"scale_{scale}_{i}.parquet").open("wb") as f:
                df_sample = df_arena.sample(n=10 ** (scale + 1), replace=True, random_state=scale * 10 + i)
                df_sample[["model_a", "model_b", "winner"]].to_parquet(f, index=False)


if __name__ == "__main__":
    main()
