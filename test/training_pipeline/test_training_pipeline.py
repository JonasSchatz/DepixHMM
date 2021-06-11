from unittest import TestCase


from test.utils import demo_picture_parameters
from text_depixelizer.parameters import PictureParameters
from text_depixelizer.training_pipeline.training_pipeline import create_training_data


class TestTrainingPipeline(TestCase):

    def test_create_training_data(self):
        # Arrange
        n_img = 1
        picture_parameters: PictureParameters = demo_picture_parameters

        # Act
        texts, original_images, pixelized_images, windows = create_training_data(n_img, picture_parameters)

        # Assert: Texts
        self.assertEqual(len(texts), n_img)
        self.assertTrue(texts[0], '123456789')

        # Assert: Original Images
        self.assertEqual(len(original_images), n_img)

        # Assert: Pixelized Images
        self.assertEqual(len(pixelized_images), n_img)

        # Assert: Windows
        self.assertEqual(len(windows), n_img)
        self.assertEqual(windows[0][0].window_index, 0)

    def test_create_training_data_random_offset(self):
        # Arrange
        n_img = 10
        picture_parameters: PictureParameters = PictureParameters(
            block_size=6,
            pattern=r'123456789',
            font=demo_picture_parameters.font,
            randomize_pixelization_origin_x=True
        )

        # Act
        _, _, pixelized_images, windows = create_training_data(n_img, picture_parameters)

        # Assert
        self.assertGreater(len(set([p.origin for p in pixelized_images])), 1)
