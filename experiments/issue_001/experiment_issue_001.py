import logging
import unittest
from pathlib import Path
from typing import Tuple

from PIL import ImageFont

from resources.fonts import DemoFontPaths
from text_depixelizer.depix_hmm import depix_hmm
from text_depixelizer.parameters import PictureParameters, TrainingParameters, LoggingParameters


class Issue001Experiments(unittest.TestCase):

    def test_issue_001(self):

        font_size: int = 30
        block_size: int = 10
        font_path: str = str(DemoFontPaths.arial)
        window_size: int = 1
        pattern: str = r'[a-zA-Z ]{10,15}'
        font_color: Tuple[int, int, int] = (255, 255, 255)
        background_color: Tuple[int, int, int] = (39, 48, 70)

        img_path: Path = Path(__file__).parent / 're-mosaiced.png' #'experiment_issue_001.png'

        picture_parameters: PictureParameters = PictureParameters(
            block_size=block_size,
            pattern=pattern,
            font=ImageFont.truetype(font_path, font_size),
            window_size=window_size,
            randomize_pixelization_origin_x=True,
            font_color=font_color,
            background_color=background_color
        )

        training_parameters: TrainingParameters = TrainingParameters(
            n_img_train=50000,
            n_img_test=20,
            n_clusters=500
        )

        logging_parameters: LoggingParameters = LoggingParameters(
            timer_log_level=logging.INFO,
            module_log_level=logging.DEBUG
        )

        reconstructed_string: str = depix_hmm(
            picture_parameters=picture_parameters,
            training_parameters=training_parameters,
            logging_parameters=logging_parameters,
            img_path=img_path
        )

        print(reconstructed_string)
