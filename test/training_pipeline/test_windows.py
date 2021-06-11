from typing import List, Tuple
from unittest import TestCase

from test import utils
from text_depixelizer.training_pipeline.original_image import OriginalImage
from text_depixelizer.training_pipeline.pixelized_image import PixelizationOptions, PixelizedImage, pixelize_image
from text_depixelizer.training_pipeline.windows import create_windows_from_image, Window, interval_overlap, WindowOptions


class TestWindows(TestCase):

    def test_interval_overlap(self):
        # Arrange: a, b, overlap
        data = [
            [(0, 30), (10, 20), 10],
            [(10, 20), (0, 30), 10],
            [(20, 40), (10, 30), 10],
            [(10, 30), (20, 40), 10],
            [(20, 40), (30, 50), 10],
            [(30, 50), (20, 40), 10],
            [(10, 20), (30, 40), 0]
        ]

        # Act
        result: List[int] = [interval_overlap(line[0], line[1]) for line in data]

        # Assert
        self.assertListEqual([line[2] for line in data], result)

    def test_window_creation(self):
        # Arrange
        text: str = 'Asdfjklö'
        block_size: int = 8
        offset: Tuple[int, int] = (0, 0)
        window_size: int = 4
        character_threshold: int = 0

        original_image: OriginalImage = utils.create_image(text=text)
        pixelization_options: PixelizationOptions = PixelizationOptions(block_size, offset)
        pixelized_image: PixelizedImage = pixelize_image(original_image, pixelization_options)
        window_options: WindowOptions = WindowOptions(window_size, character_threshold)

        # Act
        windows: List[Window] = create_windows_from_image(original_image, pixelized_image, window_options)

        # Assert: The first character in the first window is the first character of the original text
        self.assertEqual(windows[0].characters[0], text[0])

        # Assert: The amount of values in each window is correct, one for each tiles
        self.assertEqual(len(windows[0].values), window_size * 3 * pixelized_image.n_tiles[1])

        # Assert: The window index is correctly set
        self.assertEqual(windows[0].window_index, 0)

        # Assert: The number of windows is correct
        self.assertEqual(len(windows), pixelized_image.n_tiles[0] - window_options.window_size + 1)
