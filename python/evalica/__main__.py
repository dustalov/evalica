#!/usr/bin/env python3

# Copyright 2024-2026 Dmitry Ustalov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

__author__ = "Dmitry Ustalov"
__license__ = "Apache-2.0"

import argparse
import sys
from pathlib import Path
from typing import IO, cast

import pandas as pd

import evalica
from evalica import AlphaResult, Winner

WINNERS = {
    "left": Winner.X,
    "right": Winner.Y,
    "tie": Winner.Draw,
}


def read_csv(f: IO[str]) -> tuple[list[str], list[str], list[Winner]]:
    df_input = pd.read_csv(f, dtype=str)

    df_input["winner"] = df_input["winner"].str.lower().map(WINNERS)
    df_input = df_input[~df_input["winner"].isna()]

    xs = df_input["left"].tolist()
    ys = df_input["right"].tolist()
    ws = df_input["winner"].tolist()

    return xs, ys, ws


def read_alpha_csv(f: IO[str]) -> pd.DataFrame:
    return pd.read_csv(f, header=None, dtype=str)


def write_csv(f: IO[str], scores: pd.Series[str]) -> pd.DataFrame:
    df_output = pd.DataFrame(scores.rename("score"))
    df_output.index.name = "item"
    df_output = df_output.sort_values(by="score", ascending=False)
    df_output["rank"] = df_output["score"].rank(na_option="bottom", ascending=False).astype(int)
    df_output[["score", "rank"]].to_csv(f)
    return df_output


def write_alpha_csv(f: IO[str], result: AlphaResult) -> None:
    df_output = pd.DataFrame(
        {
            "metric": ["alpha", "observed", "expected"],
            "value": [result.alpha, result.observed, result.expected],
        },
    )
    df_output.to_csv(f, index=False)


def invoke(args: argparse.Namespace, f_in: IO[str]) -> pd.Series[str]:
    return cast("pd.Series[str]", args.algorithm(*read_csv(f_in)).scores)


def invoke_alpha(args: argparse.Namespace, f_in: IO[str]) -> AlphaResult:
    data = read_alpha_csv(f_in)
    return evalica.alpha(data, distance=args.distance)


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
    parser.add_argument("-i", "--input", help="input.csv", required=True, type=Path)
    parser.add_argument("-o", "--output", help="output.csv", type=Path)
    parser.add_argument("--version", action="version", version="Evalica v" + evalica.__version__)
    parser.set_defaults()

    subparsers = parser.add_subparsers(required=True, dest="command")

    pairwise_parser = subparsers.add_parser("pairwise", help="pairwise ranking methods")
    pairwise_subparsers = pairwise_parser.add_subparsers(required=True, dest="algorithm_name")

    for name, algorithm in ALGORITHMS.items():
        subparser = pairwise_subparsers.add_parser(name)
        subparser.set_defaults(func=invoke, algorithm=algorithm)

    alpha_parser = subparsers.add_parser("alpha", help="Krippendorff's alpha")
    alpha_parser.add_argument(
        "-d",
        "--distance",
        choices=["nominal", "ordinal", "interval", "ratio"],
        default="nominal",
        help="distance metric (default: nominal)",
    )
    alpha_parser.set_defaults(func=invoke_alpha)

    args = parser.parse_args()

    with args.input.open(encoding="UTF-8") as f_in:
        result = args.func(args, f_in)

    if hasattr(args, "algorithm"):
        scores = cast("pd.Series[str]", result)
        if args.output:
            with args.output.open("w", encoding="UTF-8") as f_out:
                write_csv(f_out, scores)
        else:
            write_csv(sys.stdout, scores)
    else:
        alpha_result = cast("AlphaResult", result)
        if args.output:
            with args.output.open("w", encoding="UTF-8") as f_out:
                write_alpha_csv(f_out, alpha_result)
        else:
            write_alpha_csv(sys.stdout, alpha_result)


if __name__ == "__main__":
    main()
