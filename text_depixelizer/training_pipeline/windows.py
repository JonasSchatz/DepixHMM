from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np

from text_depixelizer.training_pipeline.original_image import OriginalImage
from text_depixelizer.training_pipeline.pixelized_image import PixelizedImage


@dataclass
class Window:
    characters: Tuple[str, ...]
    values: np.ndarray
    window_index: int
    k: Optional[int] = None


@dataclass
class WindowOptions:
    window_size: int
    character_threshold: int = 0


def interval_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """
    Calculate the overlap between two intervals
    Example: a=(10, 30) and b=(20, 40) gives an overlap of 10
    """
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def create_windows_from_image(original_image: OriginalImage, pixelized_image: PixelizedImage, window_options: WindowOptions) -> List[Window]:
    windows: List[Window] = []
    block_size: int = pixelized_image.block_size

    window_width: int = window_options.window_size*block_size

    for window_index in range(pixelized_image.n_tiles[0] - window_options.window_size + 1):
        window_left: int = pixelized_image.origin[0] + window_index*block_size
        window_right: int = window_left + window_width - 1
        window_top: int = pixelized_image.origin[1]
        window_bottom: int = window_top + pixelized_image.n_tiles[1]*block_size - 1

        characters: Tuple[str, ...] = tuple(
            cbb.char for cbb in original_image.character_bounding_boxes if
            interval_overlap((cbb.left, cbb.right), (window_left, window_right)) > window_options.character_threshold
        )

        values: np.array = \
            np.asarray(
                pixelized_image.image
            )[
                window_top:window_bottom:block_size,
                window_left:window_right:block_size,
                :
            ].flatten()

        window: Window = Window(characters, values, window_index)
        windows.append(window)

    return windows
