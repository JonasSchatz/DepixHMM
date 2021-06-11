import logging
import math
import time
from collections import Counter
from typing import List, Tuple, Set

import numpy as np
from PIL import Image

from text_depixelizer.HMM.clusterer import KmeansClusterer, Clusterer
from text_depixelizer.HMM.hmm import HMM
from text_depixelizer.HMM.hmm_result_reconstructor import reconstruct_string_from_window_characters, string_similarity
from text_depixelizer.parameters import PictureParameters, TrainingParameters
from text_depixelizer.training_pipeline.training_pipeline import create_training_data
from text_depixelizer.training_pipeline.windows import Window


class DepixHMM(HMM):
    observations: List[int]
    states: List[Tuple[str, ...]]

    picture_parameters: PictureParameters
    training_parameters: TrainingParameters
    clusterer: Clusterer

    def __init__(self, picture_parameters: PictureParameters, training_parameters: TrainingParameters):
        self.picture_parameters = picture_parameters
        self.training_parameters = training_parameters

    def train(self):
        time_logger: logging.Logger = logging.getLogger('time_logger')

        # Generate training data
        texts_train, original_images_train, pixelized_images_train, windows_train = create_training_data(
            n_img=self.training_parameters.n_img_train,
            picture_parameters=self.picture_parameters
        )
        windows_train_flattened = [window for windows in windows_train for window in windows]

        t: float = time.perf_counter()
        clusterer: KmeansClusterer = KmeansClusterer(windows_train_flattened, self.training_parameters.n_clusters)
        self.clusterer = clusterer
        windows_train_flattened = clusterer.map_windows_to_cluster(windows_train_flattened)

        # ToDo: Debug
        pass

        time_logger.info(f'Performed clustering in {time.perf_counter() - t} seconds')

        used_clusters_in_training_set: int = len(set([window.k for window in windows_train_flattened]))
        if used_clusters_in_training_set != self.training_parameters.n_clusters:
            logging.error(f'\n Out of possibly {self.training_parameters.n_clusters}, only '
                          f'{used_clusters_in_training_set} are used. This might be the case when using a monospaced'
                          f'font with a font size that is a multiple of the window size.')

        # Generate observations and states
        self.calculate_hmm_properties(windows_train_flattened)

    def calculate_hmm_properties(self, windows_train: List[Window]):
        """
        Takes a flattened list of windows to determine the probability matrices of the hidden markov model
        Note that the windows have to be clustered already!
        """
        time_logger: logging.Logger = logging.getLogger('time_logger')
        t = time.perf_counter()

        observations: List[int] = list({window.k for window in windows_train})
        self.observations: List[int] = observations

        states: List[Tuple[str, ...]] = list({window.characters for window in windows_train})
        self.states: List[Tuple[str, ...]] = states

        # Compute the probability matrices
        self.starting_probabilities: np.ndarray = self.get_starting_probabilities(windows_train, states)
        self.transition_probabilities: np.ndarray = self.get_transition_probabilities(windows_train, states)
        self.emission_probabilities: np.ndarray = self.get_emission_probabilities(windows_train, states, observations)

        time_logger.info(f'Calculated HMM Properties in {time.perf_counter() - t} seconds')

    def test_image(self, img: Image):
        """
        Takes a pixelized image and reconstructs the hidden string
        """

        block_size = self.picture_parameters.block_size
        window_size = self.picture_parameters.window_size
        window_width = block_size * window_size

        n_tiles: Tuple[int] = tuple(int(el/block_size) for el in img.size)
        pixel_values_of_windows: List[np.array] = []
        for window_index in range(n_tiles[0] - window_size + 1):
            window_top = 0
            window_bottom = n_tiles[1]*block_size - 1
            window_left: int = window_index*block_size
            window_right: int = window_left + window_width - 1

            values: np.array = np.asarray(img)[
                               window_top:window_bottom:block_size,
                               window_left:window_right:block_size,
                               :].flatten()
            pixel_values_of_windows.append(values)

        k_values: List[int] = self.clusterer.map_values_to_cluster(pixel_values_of_windows)
        return self.test_cluster_indices(k_values)

    def test_windows(self, windows: List[Window]) -> str:
        """
        Takes a list of clustered windows and returns the most likely sequence of characters
        """
        windows = self.clusterer.map_windows_to_cluster(windows)
        return self.test_cluster_indices([window.k for window in windows])

    def test_cluster_indices(self, indices: List[int]):
        result: List[Tuple[str, ...]] = self.log_viterbi(indices)
        return reconstruct_string_from_window_characters(result, self.picture_parameters.block_size,
                                                         self.picture_parameters.font)


    def evaluate(self) -> Tuple[float, float]:
        """
        Generates test data and checks it with the already trained model. Returns two values:
        - Accuracy: Percentage of correctly reconstructing the string from the image
        - average_similarity: Average modified edit distance of the reconstructed string to the original text
        """

        self.print_states()

        time_logger: logging.Logger = logging.getLogger('time_logger')
        t = time.perf_counter()

        texts_evaluate, original_images_evaluate, pixelized_images_evaluate, windows_evaluate = create_training_data(
            n_img=self.training_parameters.n_img_test,
            picture_parameters=self.picture_parameters
        )

        similarities: List[float] = []
        for text, windows in zip(texts_evaluate, windows_evaluate):
            reconstructed_text: str = self.test_windows(windows)
            similarity: float = string_similarity(text, reconstructed_text)
            similarities.append(similarity)

            logging.debug(f'Expected: {text}, Actual: {reconstructed_text}, Similarity: {similarity}')

        accuracy: float = similarities.count(1.0) / len(similarities)
        average_similarity: float = sum(similarities) / len(similarities)
        time_logger.info(f'Performed Evaluation in {time.perf_counter() - t} seconds')

        return accuracy, average_similarity

    @staticmethod
    def get_starting_probabilities(windows: List[Window], states: List[Tuple[str, ...]]) -> np.ndarray:
        """
        Calculate the probability of starting in state X
        """
        starting_states_unnormalized: Counter = Counter(
            [window.characters for window in windows if window.window_index == 0])
        total: int = sum(starting_states_unnormalized.values())
        starting_probabilities: List[float] = [starting_states_unnormalized.get(state, 0) / total for state in states]
        return np.array(starting_probabilities)

    @staticmethod
    def get_transition_probabilities(windows: List[Window], states: List[Tuple[str, ...]]) -> np.ndarray:
        """
        From the given windows, count how many times state X follows state Y and save the (row-wise) normalized sum
        in transition_probabilities[X, Y]
        """
        transition_probabilities_unnormalized: np.ndarray = np.zeros((len(states), len(states)))

        for current_window, next_window in zip(windows[:-1], windows[1:]):

            # no transition inbetween images
            if next_window.window_index == 0:
                continue
            index_of_current_state: int = states.index(current_window.characters)
            index_of_next_state: int = states.index(next_window.characters)
            transition_probabilities_unnormalized[index_of_current_state, index_of_next_state] += 1

        # Normalization: If there is 0/0 (no transition leaving state X was observed in the training data), we
        # assume that every transition from that state is equally likely
        transition_probabilities: np.ndarray = np.divide(
            transition_probabilities_unnormalized,
            transition_probabilities_unnormalized.sum(axis=1)[:, np.newaxis],
            out=np.full(shape=transition_probabilities_unnormalized.shape, fill_value=1.0/len(states), dtype=float),
            where=transition_probabilities_unnormalized.sum(axis=1)[:, np.newaxis] != 0
        )
        return transition_probabilities

    @staticmethod
    def get_emission_probabilities(windows: List[Window], states: List[Tuple[str, ...]],
                                   observations: List[int]) -> np.ndarray:
        """
        Calculate the probability that state X emits symbol Y and save the (row-wise) normalized sum
        in emission_probabilities[X, Y]
        """
        emission_probabilities_unnormalized: np.ndarray = np.zeros((len(states), len(observations)))

        for window in windows:
            index_of_state = states.index(window.characters)
            index_of_observation = observations.index(window.k)
            emission_probabilities_unnormalized[index_of_state, index_of_observation] += 1

        emission_probabilities: np.ndarray = emission_probabilities_unnormalized / emission_probabilities_unnormalized.sum(axis=1)[:, np.newaxis]

        return emission_probabilities

    def print_states(self):
        unique_characters: Set[str] = set([c for char in self.states for c in char])
        max_state_length: int = max([len(state) for state in self.states])

        for i in range(1, max_state_length+1):
            states_with_length_i = [state for state in self.states if len(state) == i]
            logging.warning(f'Found {len(states_with_length_i)} states with length {i}, expected {math.pow(len(unique_characters), i)}')