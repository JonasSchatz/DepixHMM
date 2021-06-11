import random
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

from resources.fonts import DemoFontPaths
from text_depixelizer.parameters import PictureParameters, TrainingParameters
from text_depixelizer.training_pipeline.original_image import ImageCreationOptions, OriginalImage, generate_image_from_text


def create_random_mosaic(img_size: Tuple[int, int], block_size: int):
    """
    Create an image of size img_size that consists of blocks of size block_size.
    It is assured that all blocks have a different color
    """
    n_tiles = (int(img_size[0] / block_size), int(img_size[1] / block_size))

    img: Image = Image.new('RGB', img_size, (255, 255, 255))
    draw: ImageDraw = ImageDraw.Draw(img)

    for i in range(n_tiles[0]):
        for j in range(n_tiles[1]):
            left: int = i * block_size
            right: int = left + block_size - 1
            top: int = j * block_size
            bottom: int = top + block_size - 1

            draw.rectangle(
                (left, top, right, bottom),
                fill=(int(255 / img_size[0] * i), int(255 / img_size[1] * j), random.randint(0, 255))
            )

    return img


def create_image(text: str, padding: Tuple[int, int] = (30, 30), font_size: int = 50) -> OriginalImage:
    default_font: ImageFont = ImageFont.truetype(str(DemoFontPaths.arial), font_size)
    options: ImageCreationOptions = ImageCreationOptions(padding, default_font)
    return generate_image_from_text(text, options)


demo_picture_parameters: PictureParameters = PictureParameters(
    block_size=6,
    pattern=r'123456789',
    font=ImageFont.truetype(str(DemoFontPaths.arial), 50)
)

demo_training_parameters: TrainingParameters = TrainingParameters(
    n_img_train=7,
    n_img_test=3,
    n_clusters=3
)