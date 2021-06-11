from typing import Tuple, List
from unittest import TestCase

import numpy as np
from PIL import Image, ImageFont

from resources.fonts import DemoFontPaths
from test import utils
from test.utils import create_random_mosaic
from text_depixelizer.training_pipeline.original_image import ImageCreationOptions, OriginalImage, generate_image_from_text
from text_depixelizer.training_pipeline.pixelized_image import determine_number_of_tiles, PixelizationOptions, pixelize_image, \
    pixelize_area, PixelizedImage, determine_origin


class TestMosaicImage(TestCase):
    default_font_size: int = 50
    default_font: ImageFont = ImageFont.truetype(str(DemoFontPaths.arial), default_font_size)
    default_padding: Tuple[int, int] = (30, 30)

    def test_determine_number_of_tiles_no_offset(self):
        # Arrange
        text_width: int = 25
        font_metrics: Tuple[int, int] = (12, 8)
        offset: Tuple[int, int] = (0, 0)
        block_size: int = 10

        # Act
        n_tiles: Tuple[int, int] = determine_number_of_tiles(text_width, font_metrics, offset, block_size)

        # Assert
        self.assertTupleEqual((3, 3), n_tiles)

    def test_determine_number_of_tiles_y_offset_small(self):
        # Arrange
        text_width: int = 25
        font_metrics: Tuple[int, int] = (12, 8)
        offset: Tuple[int, int] = (0, 4)
        block_size: int = 10

        # Act
        n_tiles: Tuple[int, int] = determine_number_of_tiles(text_width, font_metrics, offset, block_size)

        # Assert
        self.assertTupleEqual((3, 3), n_tiles)

    def test_determine_number_of_tiles_y_offset_large(self):
        # Arrange
        text_width: int = 25
        font_metrics: Tuple[int, int] = (12, 8)
        offset: Tuple[int, int] = (0, 9)
        block_size: int = 10

        # Act
        n_tiles: Tuple[int, int] = determine_number_of_tiles(text_width, font_metrics, offset, block_size)

        # Assert
        self.assertTupleEqual((3, 3), n_tiles)

    def test_determine_origin_no_offset(self):
        # Arrange
        padding: Tuple[int, int] = (20, 20)
        font_metrics: Tuple[int, int] = (12, 8)
        offset: Tuple[int, int] = (0, 0)
        block_size: int = 10

        # Act
        origin: Tuple[int, int] = determine_origin(padding, font_metrics, offset, block_size)

        # Assert
        self.assertTupleEqual((20, 12), origin)

    def test_determine_origin_y_offset_small(self):
        # Arrange
        padding: Tuple[int, int] = (20, 20)
        font_metrics: Tuple[int, int] = (12, 8)
        offset: Tuple[int, int] = (0, 6)
        block_size: int = 10

        # Act
        origin: Tuple[int, int] = determine_origin(padding, font_metrics, offset, block_size)

        # Assert
        self.assertTupleEqual((20, 16), origin)

    def test_pixelize_image(self):
        # Arrange
        image_creation_options: ImageCreationOptions = ImageCreationOptions(self.default_padding, self.default_font)
        original_image: OriginalImage = generate_image_from_text(text='123456789', options=image_creation_options)
        block_size: int = 10
        offset: Tuple[int, int] = (0, 0)
        pixelization_options: PixelizationOptions = PixelizationOptions(block_size, offset)

        # Act
        pixelized_image: PixelizedImage = pixelize_image(original_image, pixelization_options)

        # Assert
        self.assertTrue(pixelized_image.n_tiles[0] > 0)
        self.assertEqual(pixelized_image.block_size, block_size)
        self.assertEqual(pixelized_image.origin, (30, 26))

    def test_pixelize_image_correct_offset(self):
        # Arrange
        img_size = (120, 120)
        block_size = 10

        n_tiles = (int(img_size[0] / block_size), int(img_size[1] / block_size))
        test_image: Image = create_random_mosaic(img_size, block_size)

        # Act
        pixelized_image: Image = pixelize_area(test_image, block_size, (0, 0), n_tiles)

        # Assert: Images are equal
        self.assertEqual(test_image, pixelized_image)

        # Assert: Pixel-size is correct
        self.assertNotEqual(pixelized_image.getpixel((0, 0)), pixelized_image.getpixel((block_size, 0)))
        self.assertEqual(pixelized_image.getpixel((0, 0)), pixelized_image.getpixel((block_size-1, 0)))

    def test_randomize_offset_x(self):
        # Arrange
        original_image: OriginalImage = utils.create_image(text='123456789')

        block_size: int = 10
        pixelization_options: List[PixelizationOptions] = [
            PixelizationOptions(block_size=block_size, offset=(i, 0)) for i in range(block_size+1)
        ]

        # Act
        pixelized_images: List[PixelizedImage] = [pixelize_image(original_image=original_image, pixelization_options=p) for p in pixelization_options]

        # Assert: There is a difference between images with an offset of one
        self.assertNotEqual(np.sum(np.asarray(pixelized_images[0].image) - np.asarray(pixelized_images[1].image)), 1)

        # Assert: There is no difference between images with an offset of block_size
        self.assertEqual(np.sum(np.asarray(pixelized_images[0].image) - np.asarray(pixelized_images[block_size].image)), 0)

    def test_randomize_offset_y(self):
        # Arrange
        original_image: OriginalImage = utils.create_image(text='123456789')

        block_size: int = 10
        pixelization_options: List[PixelizationOptions] = [
            PixelizationOptions(block_size=block_size, offset=(0, i)) for i in range(block_size + 1)
        ]

        # Act
        pixelized_images: List[PixelizedImage] = [pixelize_image(original_image=original_image, pixelization_options=p) for p in pixelization_options]

        # Assert: There is a difference between images with an offset of one
        self.assertNotEqual(np.sum(np.asarray(pixelized_images[0].image) - np.asarray(pixelized_images[1].image)), 1)

        # Assert: There is no difference between images with an offset of block_size
        self.assertEqual(np.sum(np.asarray(pixelized_images[0].image) - np.asarray(pixelized_images[block_size].image)), 0)

