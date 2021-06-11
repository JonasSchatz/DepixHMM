import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw

from text_depixelizer.training_pipeline.original_image import OriginalImage


@dataclass
class PixelizationOptions:
    block_size: int
    offset: Tuple[int, int]


@dataclass
class PixelizedImage:
    n_tiles: Tuple[int, int]
    block_size: int
    origin: Tuple[int, int]
    image: Image


def determine_number_of_tiles(text_width, font_metrics, offset: Tuple[int, int], block_size: int) -> Tuple[int, int]:
    tiles_y_above_baseline: int = math.ceil((font_metrics[0] - offset[1]) / block_size)
    tiles_y_below_baseline: int = math.ceil((font_metrics[1] + offset[1]) / block_size)
    tiles_x = math.ceil((text_width + offset[0]) / block_size)
    return tiles_x, tiles_y_above_baseline + tiles_y_below_baseline


def determine_origin(padding: Tuple[int, int], font_metrics: Tuple[int, int], offset: Tuple[int, int], block_size:  int) -> Tuple[int, int]:
    origin_x: int = padding[0] - offset[0]
    tiles_y_above_baseline: int = math.ceil((font_metrics[0] - offset[1]) / block_size)
    origin_y: int = padding[1] + font_metrics[0] - (offset[1] + tiles_y_above_baseline*block_size)
    return origin_x, origin_y


def get_average_color(img: Image) -> Tuple[int, int, int]:
    return tuple(np.rint(np.mean(img, axis=(0, 1))).astype(int))


def pixelize_image(original_image: OriginalImage, pixelization_options: PixelizationOptions) -> Image:
    n_tiles: Tuple[int, int] = determine_number_of_tiles(
        text_width=original_image.text_size[0],
        font_metrics=original_image.font_metrics,
        offset=(pixelization_options.offset[0] % pixelization_options.block_size, pixelization_options.offset[1] % pixelization_options.block_size),
        block_size=pixelization_options.block_size
    )

    origin: Tuple[int, int] = determine_origin(
        padding=original_image.image_creation_options.padding,
        font_metrics=original_image.font_metrics,
        offset=(pixelization_options.offset[0] % pixelization_options.block_size, pixelization_options.offset[1] % pixelization_options.block_size),
        block_size=pixelization_options.block_size
    )

    pixelized_image: Image = pixelize_area(
        image=original_image.img,
        block_size=pixelization_options.block_size,
        origin=origin,
        n_tiles=n_tiles
    )

    return PixelizedImage(
        n_tiles=n_tiles,
        block_size=pixelization_options.block_size,
        origin=origin,
        image=pixelized_image
    )


def pixelize_area(image: Image, block_size: int, origin: Tuple[int, int], n_tiles: Tuple[int, int]) -> Image:
    """
    Pixelize an area of an image, given the parameters
    """

    pixelized_image: Image = image.copy()
    draw = ImageDraw.Draw(pixelized_image)
    for i in range(n_tiles[0]):
        for j in range(n_tiles[1]):
            left: int = origin[0] + i*block_size
            right: int = left + block_size - 1
            top: int = origin[1] + j*block_size
            bottom: int = top + block_size - 1
            draw.rectangle((left, top, right, bottom), fill=get_average_color(image.crop((left, top, right+1, bottom+1))))
    return pixelized_image
