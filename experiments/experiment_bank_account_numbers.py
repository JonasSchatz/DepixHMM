import logging
import unittest

from PIL import FreeTypeFont

from resources.fonts import DemoFontPaths
from text_depixelizer.depix_hmm import depix_hmm
from text_depixelizer.parameters import PictureParameters, TrainingParameters, LoggingParameters


class BackAccountExperiments(unittest.TestCase):

    def test_bank_account_experiment_original(self):
        """
        Repeating the experiments from Ch. 3.3 of the original publication.
        It is one of the simplest tasks: Redacted text consists of exactly 7 evenly spaced digits
        """

        picture_parameters: PictureParameters = PictureParameters(
            block_size=6,
            pattern=r'\d{7}',
            font=FreeTypeFont.truetype(str(DemoFontPaths.arial), 24),
            window_size=2,
            randomize_pixelization_origin_x=True
        )

        training_parameters: TrainingParameters = TrainingParameters(
            n_img_train=10000,
            n_img_test=20,
            n_clusters=300
        )

        logging_parameters: LoggingParameters = LoggingParameters(
            timer_log_level=logging.INFO,
            module_log_level=logging.DEBUG
        )

        depix_hmm(
            picture_parameters=picture_parameters,
            training_parameters=training_parameters,
            logging_parameters=logging_parameters
        )
