import itertools
import logging
from pathlib import Path
from typing import Optional

from PIL import ImageFont, Image

from resources.fonts import DemoFontPaths
from text_depixelizer.HMM.depix_hmm import DepixHMM
from text_depixelizer.parameters import PictureParameters, TrainingParameters, LoggingParameters, \
    PictureParametersGridSearch, TrainingParametersGridSearch


def init_logging(logging_parameters: LoggingParameters):
    logging.basicConfig(level=logging_parameters.module_log_level)
    time_logger: logging.Logger = logging.getLogger('time_logger')
    time_logger.setLevel(logging_parameters.timer_log_level)


def depix_hmm(picture_parameters: PictureParameters,
              training_parameters: TrainingParameters,
              logging_parameters: LoggingParameters = None,
              img_path: Path = None) -> Optional[str]:

    if logging_parameters:
        init_logging(logging_parameters)

    # Train and evaluate the HMM
    hmm: DepixHMM = DepixHMM(picture_parameters, training_parameters)
    hmm.train()
    accuracy, average_distance = hmm.evaluate()
    logging.info(f'Accuracy: {accuracy}, Avg. Distance: {average_distance}')

    # If a path to an image was given, analyze the image
    if img_path:
        with Image.open(img_path) as img:
            reconstructed_string: str = hmm.test_image(img)
            return reconstructed_string

    return None


def depix_hmm_grid_search(picture_parameters_grid_search: PictureParametersGridSearch,
                          training_parameters_grid_search: TrainingParametersGridSearch,
                          logging_parameters: LoggingParameters = None,
                          img_path: Path = None) -> Optional[str]:
    if logging_parameters:
        init_logging(logging_parameters)

    best_hmm: Optional[DepixHMM] = None
    best_accuracy: float = 0.0
    best_avg_distance: float = 1.0

    # Iterate through grid and find best
    for window_size, n_clusters, n_img_train, offset_y in itertools.product(
            *[picture_parameters_grid_search.window_size,
              training_parameters_grid_search.n_clusters,
              training_parameters_grid_search.n_img_train,
              picture_parameters_grid_search.offset_y]):

        picture_parameters: PictureParameters = PictureParameters(
            pattern=picture_parameters_grid_search.pattern,
            font=picture_parameters_grid_search.font,
            block_size=picture_parameters_grid_search.block_size,
            window_size=window_size,
            offset_y=offset_y
        )

        training_parameters: TrainingParameters = TrainingParameters(
            n_img_test=training_parameters_grid_search.n_img_test,
            n_img_train=n_img_train,
            n_clusters=n_clusters
        )

        hmm: DepixHMM = DepixHMM(picture_parameters, training_parameters)
        hmm.train()
        accuracy, average_distance = hmm.evaluate()
        logging.info(f'Window Size: {window_size}, Clusters: {n_clusters}, Training Images: {n_img_train}, Offset Y: {offset_y}')
        logging.info(f'Accuracy: {accuracy}, Avg. Distance: {average_distance} \n')

        if img_path:
            with Image.open(img_path) as img:
                reconstructed_string: str = hmm.test_image(img)
                logging.warning(f'Reconstructed string: {reconstructed_string}')

        if accuracy > best_accuracy:
            best_hmm = hmm
            best_accuracy = accuracy
            best_avg_distance = average_distance

    # Finalize
    logging.warning(f'Found HMM with accuracy {best_accuracy} and average distance {best_avg_distance}')
    logging.warning(f'Associated parameters: ')
    logging.warning(f'    Window Size: {best_hmm.picture_parameters.window_size}')
    logging.warning(f'    Clusters: {best_hmm.training_parameters.n_clusters}')
    logging.warning(f'    Training Images: {best_hmm.training_parameters.n_img_train}')

    # If a path to an image was given, analyze the image
    if img_path:
        with Image.open(img_path) as img:
            reconstructed_string: str = best_hmm.test_image(img)
            return reconstructed_string

    return None


if __name__ == '__main__':
    image_path: Path = Path(__file__).parent.parent / 'resources' / 'images' / 'arial_50' / '123456789_blocksize-6.PNG'

    picture_parameters: PictureParameters = PictureParameters(
        pattern=r'\d{8,12}',
        font=ImageFont.truetype(str(DemoFontPaths.arial), 50),
        block_size=6
    )

    training_parameters: TrainingParameters = TrainingParameters(
        n_img_train=1000,
        n_img_test=100,
        n_clusters=300
    )

    depix_hmm(picture_parameters=picture_parameters, training_parameters=training_parameters, img_path=image_path)

