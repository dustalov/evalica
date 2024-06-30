from .evalica import (
    __version__,
    Status,
    py_matrices as matrices,
    py_counting as counting,
    py_bradley_terry as bradley_terry,
    py_newman as newman,
    py_elo as elo,
)

STATUSES = [
    evalica.Status.Won,
    evalica.Status.Lost,
    evalica.Status.Tied,
    evalica.Status.Skipped
]

__all__ = ["__version__", "Status", "matrices", "counting", "bradley_terry", "newman", "elo", "STATUSES"]
