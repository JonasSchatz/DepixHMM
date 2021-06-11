import logging
from pathlib import Path
from typing import Optional

from PIL import ImageFont

from resources.fonts import DemoFontPaths
from text_depixelizer.depix_hmm import depix_hmm_grid_search
from text_depixelizer.parameters import PictureParametersGridSearch, TrainingParametersGridSearch, LoggingParameters
from text_depixelizer.preprocessing import show_font_metrics

picture_parameters: PictureParametersGridSearch = PictureParametersGridSearch(
    pattern=r'\d{9}',
    window_size=[3],
    randomize_pixelization_origin_x=True,
    font=ImageFont.truetype(str(DemoFontPaths.arial), 50),
    block_size=8,
    offset_y=[0, 1, 2, 3, 4, 5, 6, 7]
)

#show_font_metrics(picture_parameters)

training_parameters: TrainingParametersGridSearch = TrainingParametersGridSearch(
    n_clusters=[250],
    n_img_test=1000,
    n_img_train=[10000]
)

logging_parameters: LoggingParameters = LoggingParameters(
    module_log_level=logging.INFO,
    timer_log_level=logging.INFO
)

img_path: Path = Path(__file__).parent / 'pixelized_cropped.png'

reconstructed_string: Optional[str] = depix_hmm_grid_search(
    picture_parameters_grid_search=picture_parameters,
    training_parameters_grid_search=training_parameters,
    logging_parameters=logging_parameters,
    img_path=img_path)

print(reconstructed_string)
