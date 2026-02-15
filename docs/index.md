# Evalica

**Evalica** [&#x025b;&#x02c8;&#x028b;alit&#x0361;sa] (eh-vah-lee-tsah) is an evaluation toolkit for statistical analysis, combining fast Rust implementations with Python APIs for ranking, reliability, and uncertainty estimation. Evalica is fully compatible with [NumPy](https://numpy.org/) arrays and [pandas](https://pandas.pydata.org/) data frames.

[![Evalica](https://raw.githubusercontent.com/dustalov/evalica/master/Evalica.svg)](https://github.com/dustalov/evalica)

## Installation

- [pip](https://pip.pypa.io/): `pip install evalica` [![PyPI Version][pypi_badge]][pypi_link]
- [Anaconda](https://docs.conda.io/en/latest/): `conda install conda-forge::evalica` [![Anaconda.org][conda_badge]][conda_link]
- [Cargo](https://crates.io/crates/evalica): `cargo add evalica` [![crates.io][crates_badge]][crates_link]

[pypi_badge]: https://badge.fury.io/py/evalica.svg
[pypi_link]: https://pypi.python.org/pypi/evalica
[conda_badge]: https://anaconda.org/conda-forge/evalica/badges/version.svg
[conda_link]: https://anaconda.org/conda-forge/evalica
[crates_badge]: https://img.shields.io/crates/v/evalica
[crates_link]: https://crates.io/crates/evalica

## Usage

Imagine that we would like to rank the different meals and have the following dataset of three comparisons produced by food experts.

| **Item X**| **Item Y** | **Winner** |
|:---:|:---:|:---:|
| `pizza` | `burger` | `x` |
| `burger` | `sushi` | `y` |
| `pizza` | `sushi` | `tie` |

Given this hypothetical example, Evalica takes these three columns and computes the outcome of the given pairwise comparison according to the chosen model. Note that the first argument is the column `Item X`, the second argument is the column `Item Y`, and the third argument corresponds to the column `Winner`.

```pycon
>>> from evalica import elo, Winner
>>> result = elo(
...     ['pizza', 'burger', 'pizza'],
...     ['burger', 'sushi', 'sushi'],
...     [Winner.X, Winner.Y, Winner.Draw],
... )
>>> result.scores
pizza     1014.972058
burger     970.647200
sushi     1014.380742
Name: elo, dtype: float64
```

As a result, we obtain [Elo scores](https://en.wikipedia.org/wiki/Elo_rating_system) of our items. In this example, `pizza` was the most favoured item, `sushi` was the runner-up, and `burger` was the least preferred item.

| **Item**| **Score** |
|---|---:|
| `pizza` | 1014.97 |
| `burger` | 970.65 |
| `sushi` | 1014.38 |

Evalica also supports computing [Krippendorff's alpha](https://en.wikipedia.org/wiki/Krippendorff%27s_alpha), a statistical measure of inter-rater reliability. Unlike pairwise comparisons, alpha accepts a matrix where rows represent raters (observers) and columns represent units (items being rated).

```pycon
>>> import pandas as pd
>>> from evalica import alpha
>>> data = pd.DataFrame([
...     [1, 1, None, 1],
...     [2, 2, 3, 2],
...     [3, 3, 3, 3],
...     [3, 3, 3, 3],
...     [2, 2, 2, 2],
...     [1, 2, 3, 4],
...     [4, 4, 4, 4],
...     [1, 1, 2, 1],
...     [2, 2, 2, 2],
...     [None, 5, 5, 5],
...     [None, None, 1, 1],
... ]).T
>>> result = alpha(data, distance='nominal')
>>> result.alpha
0.7434210526315788
```

This example demonstrates computing alpha with nominal distance for categorical ratings. The result indicates substantial agreement among raters (alpha ≈ 0.74). Evalica supports multiple distance metrics: `nominal`, `ordinal`, `interval`, `ratio`, or custom distance functions.

## Command-Line Interface

Evalica also provides a simple command-line interface, allowing the use of these methods in shell scripts and for prototyping.

```console
$ evalica -i food.csv pairwise bradley-terry
item,score,rank
Tacos,2.509025136024378,1
Sushi,1.1011561298265815,2
Burger,0.8549063627182466,3
Pasta,0.7403814336665869,4
Pizza,0.5718366915548537,5
```

Refer to the [food.csv](https://github.com/dustalov/evalica/blob/master/food.csv) file as an input example.

For Krippendorff's alpha, use a CSV file with ratings in a matrix format:

```console
$ evalica -i codings.csv alpha --distance=nominal
metric,value
alpha,0.743421052631579
observed,7.999999999999999
expected,31.179487179487182
```

Refer to the [codings.csv](https://github.com/dustalov/evalica/blob/master/codings.csv) file as an input example.

## Web Application

Evalica has a built-in [Gradio](https://www.gradio.app/) application that can be launched as `python3 -m evalica.gradio`. Please ensure that the library was installed as `pip install evalica[gradio]`.
