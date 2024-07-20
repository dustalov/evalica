#!/usr/bin/env python3

from __future__ import annotations

__author__ = "Dmitry Ustalov"
__license__ = "Apache-2.0"

import argparse
import sys
from typing import IO, cast

import pandas as pd

import evalica
from evalica import Winner

WINNERS = {
    "left": Winner.X,
    "right": Winner.Y,
    "tie": Winner.Draw,
}


def map_winner(winner: str) -> Winner:
    return WINNERS.get(winner.lower(), Winner.Ignore)


def read_csv(f: IO[str]) -> tuple[list[str], list[str], list[Winner]]:
    df_input = pd.read_csv(f, dtype=str)

    xs = df_input["left"].tolist()
    ys = df_input["right"].tolist()
    ws = df_input["winner"].apply(map_winner).tolist()  # type: ignore[arg-type]

    return xs, ys, ws


def write_csv(f: IO[str], scores: pd.Series[str]) -> pd.DataFrame:
    df_output = pd.DataFrame(scores.rename("score"))
    df_output.index.name = "item"
    df_output = df_output.sort_values(by="score", ascending=False)
    df_output["rank"] = df_output["score"].rank(na_option="bottom", ascending=False).astype(int)
    df_output[["score", "rank"]].to_csv(f)
    return df_output


def invoke(args: argparse.Namespace) -> pd.Series[str]:
    return cast("pd.Series[str]", args.algorithm(*read_csv(args.input)).scores)


ALGORITHMS = {
    "counting": evalica.counting,
    "average-win-rate": evalica.average_win_rate,
    "bradley-terry": evalica.bradley_terry,
    "elo": evalica.elo,
    "eigen": evalica.eigen,
    "pagerank": evalica.pagerank,
    "newman": evalica.newman,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalica v" + evalica.__version__)
    parser.add_argument("-i", "--input", help="input.csv", required=True,
                        type=argparse.FileType("r", encoding="UTF-8"))
    parser.add_argument("-o", "--output", help="output.csv", default=sys.stdout,
                        type=argparse.FileType("w", encoding="UTF-8"))
    parser.add_argument("--version", action="version", version="Evalica v" + evalica.__version__)
    parser.set_defaults()

    subparsers = parser.add_subparsers(required=True)

    for name, algorithm in ALGORITHMS.items():
        subparser = subparsers.add_parser(name)
        subparser.set_defaults(func=invoke, algorithm=algorithm)

    args = parser.parse_args()
    scores = args.func(args)

    write_csv(args.output, scores)


if __name__ == "__main__":
    main()
