# Evalica, your favourite evaluation toolkit

![Evalica](Evalica.svg)

[![Tests][github_tests_badge]][github_tests_link]
[![PyPI Version][pypi_badge]][pypi_link]
[![Codecov][codecov_badge]][codecov_link]

[github_tests_badge]: https://github.com/dustalov/evalica/actions/workflows/test.yml/badge.svg?branch=master
[github_tests_link]: https://github.com/dustalov/evalica/actions/workflows/test.yml
[pypi_badge]: https://badge.fury.io/py/evalica.svg
[pypi_link]: https://pypi.python.org/pypi/evalica
[codecov_badge]: https://codecov.io/gh/dustalov/evalica/branch/master/graph/badge.svg
[codecov_link]: https://codecov.io/gh/dustalov/evalica

- [Tutorial](https://dustalov.github.io/evalica/) (and [Tutorial.ipynb](Tutorial.ipynb))
- [Pair2Rank](https://huggingface.co/spaces/dustalov/pair2rank)

The logo has been created using [Recraft](https://www.recraft.ai/).

## Installation

- [pip](https://pip.pypa.io/): `pip install evalica`

## Usage

Evalica transforms pairwise comparisons into ranked lists of items. It offers convenient high-performant Rust implementations of the corresponding methods via [PyO3](https://pyo3.rs/), and additionally provides naïve Python code for most of them.

| **Item X**| **Item Y** | **Winner** |
|:---:|:---:|:---:|
| `pizza` | `burger` | `x` |
| `burger` | `sushi` | `y` |
| `pizza` | `sushi` | `tie` |

Given this hypothetical example, Evalica takes these three columns and computes the outcome of the given pairwise comparison according to the chosen model. Note that the first argument is the column `Item X`, the second argument is the column `Item Y`, and the third argument corresponds to the column `Winner`.

```python
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

As a result, we obtain scores of our items. In this example, `pizza` is the most favoured item, `sushi` is the runner-up, and `burger` is the least preferred item.

| **Item**| **Score** |
|---|---:|
| `pizza` | 1014.97 |
| `burger` | 970.65 |
| `sushi` | 1014.38 |

## Command-Line Interface

Evalica also provides a simple command-line interface, allowing the use of these methods in shell scripts and for prototyping.

```
$ evalica -i food.csv bradley-terry
item,score,rank
Tacos,0.43428827947351706,1
Sushi,0.19060105855071743,2
Burger,0.14797720376982199,3
Pasta,0.12815347866987045,4
Pizza,0.0989799795360731,5
```

Refer to the [food.csv](food.csv) file as an input example.

## Implemented Methods

| **Method** | **In Python** | **In Rust** |
|---|:---:|:---:|
| Counting | &#x274C; | &#x2705; |
| [Bradley&ndash;Terry] | &#x2705; | &#x2705; |
| [Elo] | &#x2705; | &#x2705; |
| [Eigenvalue] | &#x2705; | &#x2705; |
| [PageRank] | &#x274C; | &#x2705; |
| [Newman] | &#x2705; | &#x2705; |

[Bradley&ndash;Terry]: https://doi.org/10.2307/2334029
[Elo]: https://isbnsearch.org/isbn/9780923891275
[Eigenvalue]: https://doi.org/10.1086/228631
[PageRank]: https://doi.org/10.1016/S0169-7552(98)00110-X
[Newman]: https://jmlr.org/papers/v24/22-1086.html

## Copyright

Copyright (c) 2024 [Dmitry Ustalov](https://github.com/dustalov). See [LICENSE](LICENSE) for details.
