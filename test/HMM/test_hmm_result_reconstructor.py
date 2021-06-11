import unittest

from PIL import ImageFont

from resources.fonts import DemoFontPaths
from text_depixelizer.HMM.hmm_result_reconstructor import get_overlap, reconstruct_string_from_window_characters


class HmmResultReconstructorTests(unittest.TestCase):

    def test_reconstruct_string_from_window_characters_two_duplicates(self):
        # Arrange
        window_characters = [('8', '1'), ('8', '1'), ('8', '1'), ('8', '1'), ('8', '1'), ('1', '2'), ('1', '2'), ('1', '2'), ('1', '2'), ('1', '2'), ('2', '9'), ('2', '9'), ('2', '9'), ('2', '9'), ('9', '2'), ('9', '2'), ('9', '2'),('9', '2'), ('9', '2'), ('2', '7'), ('2', '7'), ('2', '7'), ('2', '7'), ('2', '7'), ('7', '7'), ('7', '7'), ('7', '7'), ('7', '7'), ('7', '2'), ('7', '2'), ('7', '2'), ('7', '2'), ('7', '2'), ('2', '0'), ('2', '0'),('2', '0'), ('2', '0'), ('2', '0'), ('0', '2'), ('0', '2'), ('0', '2'), ('0', '2'), ('2',)]
        expected_reconstructed_string: str = '8129277202'
        font_size: int = 50
        block_size: int = 6
        font = ImageFont.truetype(str(DemoFontPaths.arial), font_size)

        # Act
        reconstructed_string: str = reconstruct_string_from_window_characters(
            window_characters=window_characters,
            block_size=block_size,
            font=font
        )

        # Assert
        self.assertEqual(reconstructed_string, expected_reconstructed_string)

    def test_reconstruct_string_from_window_characters_seven_duplicates(self):
        # Arrange
        window_characters = [('1', '2'), ('1', '2'), ('1', '2'), ('1', '2'), ('1', '2'), ('2', '3'), ('2', '3'), ('2', '3'), ('2', '3'), ('2', '3'), ('3', '4'), ('3', '4'), ('3', '4'), ('3', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '4'), ('4', '5'), ('4', '5'), ('4', '5'), ('4', '5'), ('4', '5'), ('5',)]
        expected_reconstructed_string: str = '12344444445'
        font_size: int = 50
        block_size: int = 6
        font = ImageFont.truetype(str(DemoFontPaths.arial), font_size)

        # Act
        reconstructed_string: str = reconstruct_string_from_window_characters(
            window_characters=window_characters,
            block_size=block_size,
            font=font
        )

        # Assert
        self.assertEqual(reconstructed_string, expected_reconstructed_string)

    def test_check_overlap_complete(self):
        # Arrange
        reconstructed_data = ['1', '2', '3']
        new_characters = ('2', '3')
        expected_overlap = 2

        # Act
        actual_overlap: int = get_overlap(reconstructed_data, new_characters)

        # Assert
        self.assertEqual(actual_overlap, expected_overlap)

    def test_check_overlap_partial(self):
        # Arrange
        reconstructed_data = ['1', '2', '3']
        new_characters = ('3', '4')
        expected_overlap = 1

        # Act
        actual_overlap: int = get_overlap(reconstructed_data, new_characters)

        # Assert
        self.assertEqual(actual_overlap, expected_overlap)

    def test_check_overlap_no_overlap(self):
        # Arrange
        reconstructed_data = ['1', '2', '3']
        new_characters = ('4',)
        expected_overlap = 0

        # Act
        actual_overlap: int = get_overlap(reconstructed_data, new_characters)

        # Assert
        self.assertEqual(actual_overlap, expected_overlap)

    def test_check_overlap_empty_reconstructed_data(self):
        # Arrange
        reconstructed_data = []
        new_characters = ('4',)
        expected_overlap = 0

        # Act
        actual_overlap: int = get_overlap(reconstructed_data, new_characters)

        # Assert
        self.assertEqual(actual_overlap, expected_overlap)

    def test_check_overlap_partially_empty_reconstructed_data(self):
        # Arrange
        reconstructed_data = ['4']
        new_characters = ('4', '5')
        expected_overlap = 1

        # Act
        actual_overlap: int = get_overlap(reconstructed_data, new_characters)

        # Assert
        self.assertEqual(actual_overlap, expected_overlap)
