import logging
from unittest import TestCase

from PIL import ImageFont

from resources.fonts import DemoFontPaths
from text_depixelizer.depix_hmm import depix_hmm_grid_search
from text_depixelizer.parameters import PictureParametersGridSearch, TrainingParametersGridSearch, LoggingParameters


class TestDepixHmm(TestCase):

    def test_depix_hmm_grid_search(self):
        # Arrange
        picture_parameters: PictureParametersGridSearch = PictureParametersGridSearch(
            pattern=r'\d{8,12}',
            font=ImageFont.truetype(str(DemoFontPaths.arial), 50),
            block_size=6,
            window_size=[4, 5]
        )

        training_parameters: TrainingParametersGridSearch = TrainingParametersGridSearch(
            n_img_test=150,
            n_clusters=[50, 100],
            n_img_train=[100]
        )

        logging_parameters: LoggingParameters = LoggingParameters(
            module_log_level=logging.INFO,
            timer_log_level=logging.INFO
        )

        # Act
        depix_hmm_grid_search(picture_parameters, training_parameters, logging_parameters)

        # Assert
        pass
