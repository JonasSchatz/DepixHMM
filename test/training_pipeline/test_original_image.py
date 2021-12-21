from typing import Tuple, List
from unittest import TestCase, skip

from PIL import ImageFont, Image

from resources.fonts import DemoFontPaths
from text_depixelizer.training_pipeline.original_image import ImageCreationOptions, generate_image_from_text, \
    OriginalImage, draw_character_bounding_boxes, generate_character_bounding_boxes, CharacterBoundingBox


class TestOriginalImage(TestCase):
    default_font_size: int = 30
    default_font_color: Tuple[int, int, int] = (255, 255, 255)
    default_background_color: Tuple[int, int, int] = (0, 0, 0)
    default_font: ImageFont = ImageFont.truetype(str(DemoFontPaths.arial), default_font_size)
    default_padding: Tuple[int, int] = (30, 30)

    def test_create_image(self):
        # Arrange
        options: ImageCreationOptions = ImageCreationOptions(
            self.default_padding, self.default_font, self.default_font_color, self.default_background_color
        )
        text: str = '123456789'

        # Act
        original_image: OriginalImage = generate_image_from_text(text, options)

        # Assert: Character bounding boxes are added
        self.assertEqual(len(original_image.character_bounding_boxes), len(text))

    def test_generate_character_bounding_boxes(self):
        # Arrange
        options: ImageCreationOptions = ImageCreationOptions(self.default_padding, self.default_font)
        text: str = 'Asdf'

        # Act
        character_bounding_boxes: List[CharacterBoundingBox] = generate_character_bounding_boxes(text, options)

        # Assert
        self.assertEqual(len(character_bounding_boxes), len(text))
        self.assertEqual(character_bounding_boxes[0].left, self.default_padding[0])
        self.assertTrue(character_bounding_boxes[0].right > character_bounding_boxes[0].left)
        self.assertTrue(character_bounding_boxes[0].top >= self.default_padding[1])
        self.assertTrue(character_bounding_boxes[0].bottom <= self.default_padding[1] + self.default_font_size)

    #@skip('Only needed for visualization')
    def test_draw_character_bounding_boxes(self):
        # Arrange
        background_color: Tuple[int, int, int] = (255, 255, 255)
        font_color: Tuple[int, int, int] = (150, 0, 0)
        padding: Tuple[int, int] = (0, 0)
        options: ImageCreationOptions = ImageCreationOptions(padding, self.default_font, font_color, background_color)
        text: str = 'agagA'
        original_image: OriginalImage = generate_image_from_text(text, options)

        # Act
        image_with_bounding_boxes: Image = draw_character_bounding_boxes(original_image)

        # Assert
        image_with_bounding_boxes.show()
        pass
