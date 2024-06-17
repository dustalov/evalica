from typing import Tuple

import numpy as np
import numpy.typing as npt

def py_bradley_terry(
    m: npt.NDArray[np.int64]
) -> Tuple[int, npt.NDArray[np.float64]]: ...

def py_newman(
    m: npt.NDArray[np.int64],
    seed: int,
    tolerance: float,
    max_iter: int
) -> Tuple[int, npt.NDArray[np.float64]]: ...
