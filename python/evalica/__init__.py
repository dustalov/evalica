from .evalica import Winner, __version__
from .evalica import py_bradley_terry as bradley_terry
from .evalica import py_counting as counting
from .evalica import py_eigen as eigen
from .evalica import py_elo as elo
from .evalica import py_matrices as matrices
from .evalica import py_newman as newman
from .naive import bradley_terry as bradley_terry_naive
from .naive import newman as newman_naive

WINNERS = [
    Winner.X,
    Winner.Y,
    Winner.Draw,
    Winner.Ignore,
]

__all__ = ["__version__", "Winner", "matrices", "counting", "bradley_terry", "newman", "elo", "eigen", "WINNERS",
           "bradley_terry_naive", "newman_naive"]
