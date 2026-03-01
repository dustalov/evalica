#!/usr/bin/env python3

# Copyright 2023 Dmitry Ustalov
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
__license__ = "Apache 2.0"

import argparse
from typing import TYPE_CHECKING, Protocol, cast

try:
    import gradio as gr
except ModuleNotFoundError:
    # ModuleNotFoundError: No module named 'pyaudioop'
    import importlib.util
    import sys
    from pathlib import Path

    if (pydub_spec := importlib.util.find_spec("pydub")) is not None and (
        pydub_origin := pydub_spec.origin
    ) is not None:
        sys.path.append(str(Path(pydub_origin).parent))

    import gradio as gr

import evalica
import pandas as pd
import plotly.express as px
from evalica import Result, Winner
from plotly.graph_objects import Figure  # noqa: TC002

if TYPE_CHECKING:
    from collections.abc import Collection


def visualize(df_pairwise: pd.DataFrame) -> Figure:
    fig = px.imshow(df_pairwise, color_continuous_scale="RdBu", text_auto=".2f")
    fig.update_layout(xaxis_title="Loser", yaxis_title="Winner", xaxis_side="top")
    fig.update_traces(hovertemplate="Winner: %{y}<br>Loser: %{x}<br>Fraction of Wins: %{z}<extra></extra>")
    return fig


ALGORITHMS = {
    "Counting": evalica.counting,
    "Average Win Rate": evalica.average_win_rate,
    "Bradley-Terry": evalica.bradley_terry,
    "Elo": evalica.elo,
    "Eigenvector": evalica.eigen,
    "PageRank": evalica.pagerank,
    "Newman": evalica.newman,
}

WINNERS = {
    "left": Winner.X,
    "right": Winner.Y,
    "tie": Winner.Draw,
}


class CallableAlgorithm(Protocol):
    def __call__(
        self,
        xs: pd.Series[str],
        ys: pd.Series[str],
        winners: Collection[Winner],
    ) -> Result: ...


def invoke(
    algorithm: str,
    xs: pd.Series[str],
    ys: pd.Series[str],
    winners: Collection[Winner],
) -> pd.Series[float]:
    algorithm_impl = cast("CallableAlgorithm", ALGORITHMS[algorithm])

    return algorithm_impl(xs=xs, ys=ys, winners=winners).scores


def handler(
    file: str | None,
    algorithm: str,
    truncate: bool,  # noqa: FBT001
) -> tuple[pd.DataFrame, Figure]:
    if file is None:
        raise gr.Error("File must be uploaded")  # noqa: EM101, TRY003

    if algorithm not in ALGORITHMS:
        raise gr.Error(f"Unknown algorithm: {algorithm}")  # noqa: EM102, TRY003

    try:
        df_pairs = pd.read_csv(file, dtype=str)
    except ValueError as e:
        raise gr.Error(f"Parsing error: {e}") from e  # noqa: EM102, TRY003

    if not pd.Series(["left", "right", "winner"]).isin(df_pairs.columns).all():
        raise gr.Error("Columns must exist: left, right, winner")  # noqa: EM101, TRY003

    if not df_pairs["winner"].isin(pd.Series(["left", "right", "tie"])).all():
        raise gr.Error("Allowed winner values: left, right, tie")  # noqa: EM101, TRY003

    df_pairs = df_pairs[["left", "right", "winner"]]

    df_pairs = df_pairs.dropna(axis=0)

    df_pairs["winner"] = df_pairs["winner"].str.lower().map(WINNERS)

    scores = invoke(algorithm, df_pairs["left"], df_pairs["right"], df_pairs["winner"])

    df_result = scores.to_frame(name="score")
    df_result.index.name = "item"

    df_result["pairs"] = (
        pd.Series(0, dtype=int, index=scores.index)
        .add(
            df_pairs.groupby("left")["left"].count(),
            fill_value=0,
        )
        .add(
            df_pairs.groupby("right")["right"].count(),
            fill_value=0,
        )
        .astype(int)
    )

    df_result["rank"] = df_result["score"].rank(na_option="bottom", ascending=False).astype(int)

    if truncate:
        df_result = pd.concat([df_result.head(5), df_result.tail(5)])
        df_result = df_result[~df_result.index.duplicated(keep="last")]

    fig = visualize(evalica.pairwise_frame(df_result["score"]))

    df_result = df_result.reset_index()

    return df_result, fig


def alpha_handler(
    file: str | None,
    distance: str,
    n_resamples: int,
    confidence_level: float,
) -> pd.DataFrame:
    if file is None:
        raise gr.Error("File must be uploaded")  # noqa: EM101, TRY003

    try:
        df_ratings = pd.read_csv(file, header=None, dtype=str)
    except ValueError as e:
        raise gr.Error(f"Parsing error: {e}") from e  # noqa: EM102, TRY003

    if df_ratings.empty:
        raise gr.Error("The file is empty")  # noqa: EM101, TRY003

    try:
        result = evalica.alpha_bootstrap(
            df_ratings,
            distance=distance,  # type: ignore[arg-type]
            n_resamples=n_resamples,
            confidence_level=confidence_level,
        )
    except evalica.InsufficientRatingsError as e:
        raise gr.Error("Insufficient ratings: no units have at least 2 ratings") from e  # noqa: EM101, TRY003
    except evalica.UnknownDistanceError as e:
        raise gr.Error(f"Unknown distance: {e}") from e  # noqa: EM102, TRY003
    except Exception as e:
        raise gr.Error(f"Computation error: {e}") from e  # noqa: EM102, TRY003

    return pd.DataFrame(
        {
            "Metric": [
                "Alpha",
                "Observed Disagreement",
                "Expected Disagreement",
                f"Lower Bound ({confidence_level:.0%})",
                f"Upper Bound ({confidence_level:.0%})",
            ],
            "Value": [result.alpha, result.observed, result.expected, result.low, result.high],
        },
    )


def pairwise_interface() -> gr.Interface:
    return gr.Interface(
        fn=handler,
        inputs=[
            gr.File(
                file_types=[".tsv", ".csv"],
                label="Comparisons",
            ),
            gr.Dropdown(
                choices=list(ALGORITHMS),
                value="Bradley-Terry",
                label="Algorithm",
            ),
            gr.Checkbox(
                value=False,
                label="Truncate Output",
                info="Perform the entire computation but output only five head and five tail items, avoiding overlap.",
            ),
        ],
        outputs=[
            gr.Dataframe(
                headers=["item", "score", "pairs", "rank"],
                label="Ranking",
            ),
            gr.Plot(
                label="Win Rates",
            ),
        ],
        title="Pairwise Ranking",
        analytics_enabled=False,
        flagging_mode="never",
        fill_width=True,
    )


def alpha_interface() -> gr.Interface:
    return gr.Interface(
        fn=alpha_handler,
        inputs=[
            gr.File(
                file_types=[".tsv", ".csv"],
                label="Ratings Matrix (CSV without header)",
            ),
            gr.Dropdown(
                choices=["nominal", "ordinal", "interval", "ratio"],
                value="nominal",
                label="Distance Metric",
                info="Nominal for categorical, ordinal for ordered categories, interval/ratio for numeric scales",
            ),
            gr.Slider(
                minimum=100,
                maximum=10000,
                step=100,
                value=5000,
                label="Number of Resamples",
            ),
            gr.Slider(
                minimum=0.01,
                maximum=0.99,
                step=0.01,
                value=0.95,
                label="Confidence Level",
            ),
        ],
        outputs=[
            gr.Dataframe(
                headers=["Metric", "Value"],
                label="Inter-Rater Reliability",
            ),
        ],
        title="Krippendorff's Alpha",
        analytics_enabled=False,
        flagging_mode="never",
        fill_width=True,
    )


def interface() -> gr.TabbedInterface:
    return gr.TabbedInterface(
        [pairwise_interface(), alpha_interface()],
        ["Pairwise Ranking", "Krippendorff's Alpha"],
        title="Evalica",
        analytics_enabled=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalica v" + evalica.__version__)
    parser.add_argument(
        "--version",
        action="version",
        version=f"Evalica v{evalica.__version__} with Gradio v{gr.__version__}",
    )
    parser.add_argument("--share", action="store_true", help="create a publicly shareable link")

    args = parser.parse_args()

    interface().launch(share=args.share)


if __name__ == "__main__":
    main()
