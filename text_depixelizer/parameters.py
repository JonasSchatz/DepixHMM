from dataclasses import dataclass, field
import logging
from typing import List, Tuple

from PIL.ImageFont import FreeTypeFont


@dataclass
class PictureParameters:
    pattern: str
    font: FreeTypeFont
    font_color: Tuple[int, int, int] = (0, 0, 0)
    background_color: Tuple[int, int, int] = (255, 255, 255)
    block_size: int = None # ToDo: can be inferred
    randomize_pixelization_origin_x: bool = False
    window_size: int = 5
    offset_y: int = 0


@dataclass
class PictureParametersGridSearch(PictureParameters):
    window_size: List[int] = field(default_factory=lambda: [5])
    offset_y: List[int] = field(default_factory=lambda: [0])


@dataclass
class TrainingParameters:
    n_img_train: int
    n_img_test: int
    n_clusters: int


@dataclass
class TrainingParametersGridSearch(TrainingParameters):
    n_img_train: List[int]
    n_clusters: List[int]


@dataclass
class LoggingParameters:
    timer_log_level: int = logging.INFO
    module_log_level: int = logging.INFO
