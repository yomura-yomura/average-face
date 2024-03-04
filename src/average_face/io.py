import numpy as np
import PIL.Image

from .typing import FilePath, Image


def load_image(path: FilePath) -> Image:
    return np.array(PIL.Image.open(path))
