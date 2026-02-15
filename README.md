# Evalica, your favourite evaluation toolkit

[![Evalica](https://raw.githubusercontent.com/dustalov/evalica/master/Evalica.svg)](https://github.com/dustalov/evalica)

[![Tests][github_tests_badge]][github_tests_link]
[![Read the Docs][rtfd_badge]][rtfd_link]
[![PyPI Version][pypi_badge]][pypi_link]
[![Anaconda.org][conda_badge]][conda_link]
[![crates.io][crates_badge]][crates_link]
[![Codecov][codecov_badge]][codecov_link]
[![CodSpeed Badge][codspeed_badge]][codspeed_link]

[github_tests_badge]: https://github.com/dustalov/evalica/actions/workflows/test.yml/badge.svg?branch=master
[github_tests_link]: https://github.com/dustalov/evalica/actions/workflows/test.yml
[rtfd_badge]: https://readthedocs.org/projects/evalica/badge/
[rtfd_link]: https://evalica.readthedocs.io/
[pypi_badge]: https://badge.fury.io/py/evalica.svg
[pypi_link]: https://pypi.python.org/pypi/evalica
[conda_badge]: https://anaconda.org/conda-forge/evalica/badges/version.svg
[conda_link]: https://anaconda.org/conda-forge/evalica
[crates_badge]: https://img.shields.io/crates/v/evalica
[crates_link]: https://crates.io/crates/evalica
[codecov_badge]: https://codecov.io/gh/dustalov/evalica/branch/master/graph/badge.svg
[codecov_link]: https://codecov.io/gh/dustalov/evalica
[codspeed_badge]: https://img.shields.io/endpoint?url=https://codspeed.io/badge.json
[codspeed_link]: https://codspeed.io/dustalov/evalica

**Evalica** [&#x025b;&#x02c8;&#x028b;alit&#x0361;sa] (eh-vah-lee-tsah) is an evaluation toolkit for statistical analysis, combining fast Rust implementations with Python APIs for ranking, reliability, and uncertainty estimation. Evalica is fully compatible with [NumPy](https://numpy.org/) arrays and [pandas](https://pandas.pydata.org/) data frames.

- [Tutorial](docs/tutorial.ipynb)
- [Chatbot-Arena.ipynb](Chatbot-Arena.ipynb) [![Open in Colab][colab_badge]][colab_link] [![Binder][binder_badge]][binder_link]
- [Evalica Demo](https://huggingface.co/spaces/dustalov/evalica)

[colab_badge]: https://colab.research.google.com/assets/colab-badge.svg
[colab_link]: https://colab.research.google.com/github/dustalov/evalica/blob/master/Chatbot-Arena.ipynb
[binder_badge]: https://mybinder.org/badge_logo.svg
[binder_link]: https://mybinder.org/v2/gh/dustalov/evalica/HEAD?labpath=Chatbot-Arena.ipynb

The logo was created using [Recraft](https://www.recraft.ai/).

## Installation

- [pip](https://pip.pypa.io/): `pip install evalica`
- [Anaconda](https://docs.conda.io/en/latest/): `conda install conda-forge::evalica`
- [Cargo](https://crates.io/crates/evalica): `cargo add evalica`

## Pairwise Comparisons

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

### Inter-Rater Reliability

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

### Pairwise Ranking

```console
$ evalica -i food.csv pairwise bradley-terry
item,score,rank
Tacos,2.509025136024378,1
Sushi,1.1011561298265815,2
Burger,0.8549063627182466,3
Pasta,0.7403814336665869,4
Pizza,0.5718366915548537,5
```

Refer to the [food.csv](food.csv) file as an input example.

### Krippendorff's Alpha

For Krippendorff's alpha, use a CSV file with ratings in a matrix format (no header):

```console
$ evalica -i codings.csv alpha --distance=nominal
metric,value
alpha,0.743421052631579
observed,7.999999999999999
expected,31.179487179487182
```

## Web Application

Evalica has a built-in [Gradio](https://www.gradio.app/) application that can be launched as `python3 -m evalica.gradio`. Please ensure that the library was installed as `pip install evalica[gradio]`.

## Implemented Methods

| **Method** | **In Python** | **In Rust** |
|---|:---:|:---:|
| Counting | &#x2705; | &#x2705; |
| Average Win Rate | &#x2705; | &#x2705; |
| [Bradley&ndash;Terry] | &#x2705; | &#x2705; |
| [Elo] | &#x2705; | &#x2705; |
| [Eigenvalue] | &#x2705; | &#x2705; |
| [PageRank] | &#x2705; | &#x2705; |
| [Newman] | &#x2705; | &#x2705; |
| [Krippendorff's Alpha] | &#x2705; | &#x2705; |

<!-- Present: &#x2705; / Absent: &#x274C; -->

[Bradley&ndash;Terry]: https://doi.org/10.2307/2334029
[Elo]: https://isbnsearch.org/isbn/9780923891275
[Eigenvalue]: https://doi.org/10.1086/228631
[PageRank]: https://doi.org/10.1016/S0169-7552(98)00110-X
[Newman]: https://jmlr.org/papers/v24/22-1086.html
[Krippendorff's Alpha]: https://en.wikipedia.org/wiki/Krippendorff%27s_alpha

## Contributing

Evalica is a mixed Rust/Python project that uses [PyO3](https://pyo3.rs/), so it requires setting up the [Maturin](https://www.maturin.rs/) build system.

To set up the environment, we recommend using the [uv](https://github.com/astral-sh/uv) package manager, as demonstrated in [our test suite](.github/workflows/test.yml):

```console
$ uv venv
$ uv pip install maturin
$ source .venv/bin/activate
$ maturin develop --uv --extras dev,docs,gradio
```

In case `uv` is not available, you can use the following workaround:

```console
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install maturin
$ maturin develop --extras dev,docs,gradio
```

It is also possible to omit the Rust-accelerated routines via `pip install --no-binary evalica`.

We welcome pull requests on GitHub: <https://github.com/dustalov/evalica>. To contribute, fork the repository, create a separate branch for your changes, and submit a pull request.

## Citation

- Ustalov, D. [Reliable, Reproducible, and Really Fast Leaderboards with Evalica](https://aclanthology.org/2025.coling-demos.6). 2025. Proceedings of the 31st International Conference on Computational Linguistics: System Demonstrations. 46&ndash;53. arXiv: [2412.11314 [cs.CL]](https://arxiv.org/abs/2412.11314).

```bibtex
@inproceedings{Ustalov:25,
  author    = {Ustalov, Dmitry},
  title     = {{Reliable, Reproducible, and Really Fast Leaderboards with Evalica}},
  year      = {2025},
  booktitle = {Proceedings of the 31st International Conference on Computational Linguistics: System Demonstrations},
  pages     = {46--53},
  address   = {Abu Dhabi, UAE},
  publisher = {Association for Computational Linguistics},
  eprint    = {2412.11314},
  eprinttype = {arxiv},
  eprintclass = {cs.CL},
  url       = {https://aclanthology.org/2025.coling-demos.6},
  language  = {english},
}
```

The code for replicating the experiments is available in the [`coling2025`](coling2025/) directory.

## Copyright

Copyright (c) 2024&ndash;2026 [Dmitry Ustalov](https://github.com/dustalov). See [LICENSE](LICENSE) for details.
