import logging
from pathlib import Path
from typing import Optional

from PIL import ImageFont

from resources.fonts import DemoFontPaths
from text_depixelizer.depix_hmm import depix_hmm
from text_depixelizer.parameters import PictureParameters, TrainingParameters, LoggingParameters
from text_depixelizer.preprocessing import show_font_metrics

picture_parameters: PictureParameters = PictureParameters(
    pattern=r'\d{9}',
    window_size=3,
    randomize_pixelization_origin_x=True,
    font=ImageFont.truetype(str(DemoFontPaths.arial), 50),
    block_size=8,
    offset_y=7
)

#show_font_metrics(picture_parameters)

training_parameters: TrainingParameters = TrainingParameters(
    n_clusters=250,
    n_img_test=500,
    n_img_train=5000
)

logging_parameters: LoggingParameters = LoggingParameters(
    module_log_level=logging.DEBUG,
    timer_log_level=logging.DEBUG
)

img_path: Path = Path(__file__).parent / 'pixelized_cropped.png'

reconstructed_string: Optional[str] = depix_hmm(
    picture_parameters=picture_parameters,
    training_parameters=training_parameters,
    logging_parameters=logging_parameters,
    img_path=img_path)

print(reconstructed_string)
