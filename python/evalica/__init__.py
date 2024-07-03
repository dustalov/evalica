from .evalica import Status, __version__
from .evalica import py_bradley_terry as bradley_terry
from .evalica import py_counting as counting
from .evalica import py_eigen as eigen
from .evalica import py_elo as elo
from .evalica import py_matrices as matrices
from .evalica import py_newman as newman

STATUSES = [
    Status.Won,
    Status.Lost,
    Status.Tied,
    Status.Skipped,
]

__all__ = ["__version__", "Status", "matrices", "counting", "bradley_terry", "newman", "elo", "eigen", "STATUSES"]
