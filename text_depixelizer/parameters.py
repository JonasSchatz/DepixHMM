from dataclasses import dataclass, field
import logging
from typing import List

from PIL import ImageFont


@dataclass
class PictureParameters:
    pattern: str
    font: ImageFont
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
