from os import PathLike
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FilePath",
    "Image",
    "Landmarks",
]

FilePath: TypeAlias = str | PathLike[str]
Image: TypeAlias = NDArray[np.uint8]  # Shape: (height, width, 3)
Landmarks: TypeAlias = NDArray[np.float64]  # Shape: (68, 2)
