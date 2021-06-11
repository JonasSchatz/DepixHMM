from typing import Tuple

from PIL import ImageDraw

from text_depixelizer.parameters import PictureParameters
from text_depixelizer.training_pipeline.original_image import OriginalImage
from text_depixelizer.training_pipeline.pixelized_image import PixelizedImage
from text_depixelizer.training_pipeline.training_pipeline import create_training_data


def show_font_metrics(picture_parameters: PictureParameters):
    # Create one training example
    texts, original_images, pixelized_images, windows = create_training_data(1, picture_parameters)

    # Extract all relevant information
    original_image: OriginalImage = original_images[0]
    pixelized_image: PixelizedImage = pixelized_images[0]

    ascent, descent = original_image.image_creation_options.font.getmetrics()
    padding = original_image.image_creation_options.padding
    width: int = original_image.img.size[0]
    n_tiles: Tuple[int, int] = pixelized_images[0].n_tiles
    block_size: int = pixelized_images[0].block_size
    pixelization_origin: Tuple[int, int] = pixelized_images[0].origin

    # Derive additional information
    baseline_y = padding[1] + ascent

    # Draw
    original_draw = ImageDraw.Draw(original_image.img)
    pixelized_draw = ImageDraw.Draw(pixelized_image.image)

    # Draw baseline
    original_draw.line((padding[0], baseline_y, width - padding[0], baseline_y), fill='red', width=1)
    pixelized_draw.line((padding[0], baseline_y, width - padding[0], baseline_y), fill='red', width=1)

    # Draw ascent and descent
    original_draw.rectangle([(padding[1], padding[0]), (width - padding[0], padding[0] + ascent + descent)], outline='blue')
    pixelized_draw.rectangle([(padding[1], padding[0]), (width - padding[0], padding[0] + ascent + descent)], outline='blue')

    # Draw pixelization bounding box
    pixelized_draw.rectangle([pixelization_origin, (pixelization_origin[0] + n_tiles[0]*block_size, pixelization_origin[1] + n_tiles[1]*block_size)], outline='green')

    # Show
    original_image.img.show()
    pixelized_image.image.show()
