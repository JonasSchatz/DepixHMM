import logging
import unittest

from PIL import ImageFont

from resources.fonts import DemoFontPaths
from text_depixelizer.depix_hmm import depix_hmm
from text_depixelizer.parameters import PictureParameters, TrainingParameters, LoggingParameters


class PipelineExperiments(unittest.TestCase):

    def test_increasing_sample_images(self):
        picture_parameters: PictureParameters = PictureParameters(
            block_size=6,
            pattern=r'\d{8,12}',
            font=ImageFont.truetype(str(DemoFontPaths.arial), 50)
        )

        training_parameters: TrainingParameters = TrainingParameters(
            n_img_train=10,
            n_img_test=3,
            n_clusters=100
        )

        logging_parameters: LoggingParameters = LoggingParameters(
            timer_log_level=logging.INFO
        )

        depix_hmm(picture_parameters=picture_parameters, training_parameters=training_parameters, logging_parameters=logging_parameters)
