import unittest
from pathlib import Path
from typing import List

import numpy as np
from PIL import ImageFont, Image

from resources.fonts import DemoFontPaths
from test.utils import demo_training_parameters, demo_picture_parameters
from text_depixelizer.HMM.depix_hmm import DepixHMM
from text_depixelizer.parameters import PictureParameters, TrainingParameters
from text_depixelizer.training_pipeline.windows import Window


class TestDepixHmm(unittest.TestCase):

    demo_picture_parameters: PictureParameters = PictureParameters(
        block_size=6,
        pattern=r'\d{8,12}',
        font=ImageFont.truetype(str(DemoFontPaths.arial), 50)
    )

    def test_train(self):
        # Arrange
        training_parameters: TrainingParameters = demo_training_parameters
        depix_hmm: DepixHMM = DepixHMM(self.demo_picture_parameters, demo_training_parameters)

        # Act
        depix_hmm.train()

        # Assert
        self.assertEqual(depix_hmm.emission_probabilities.shape[1], training_parameters.n_clusters)
        self.assertTrue(len(depix_hmm.states) > 5)
        self.assertEqual(depix_hmm.emission_probabilities.shape, depix_hmm.log_emission_probabilities.shape)

    def test_evaluate(self):
        # Arrange
        depix_hmm: DepixHMM = DepixHMM(self.demo_picture_parameters, demo_training_parameters)
        depix_hmm.train()

        # Act
        accuracy, average_distance = depix_hmm.evaluate()

        # Assert
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
        self.assertIsInstance(accuracy, float)
        self.assertIsInstance(average_distance, float)


    def test_get_starting_probabilities(self):
        # Arrange
        windows: List[Window] = [
            Window(characters=('A', 'b'), values=np.ndarray([1, 2, 3]), window_index=0, k=0),
            Window(characters=('b',), values=np.ndarray([2, 3, 4]), window_index=1, k=0),
            Window(characters=('b',), values=np.ndarray([3, 4, 5]), window_index=2, k=1),
            Window(characters=('b', 'c'), values=np.ndarray([4, 5, 6]), window_index=3, k=1),
            Window(characters=('d',), values=np.ndarray([5, 6, 7]), window_index=4, k=2),
            Window(characters=('X',), values=np.ndarray([6, 7, 8]), window_index=0, k=3)
        ]
        depix_hmm: DepixHMM = DepixHMM(demo_picture_parameters, demo_training_parameters)

        # Act
        depix_hmm.calculate_hmm_properties(windows_train=windows)

        # Assert: Observations
        self.assertCountEqual(depix_hmm.observations, (0, 1, 2, 3))

        # Assert: States
        self.assertCountEqual(depix_hmm.states, (('A', 'b'), ('b',), ('b', 'c'), ('d',), ('X',)))

        # Assert: Starting probabilities
        self.assertEqual(depix_hmm.starting_probabilities[depix_hmm.states.index(('A', 'b'))], 0.5)
        self.assertEqual(depix_hmm.starting_probabilities[depix_hmm.states.index(('b',))], 0.0)

        # Assert: Transition Probabilities
        self.assertEqual(depix_hmm.transition_probabilities.shape, (len(depix_hmm.states), len(depix_hmm.states)))
        self.assertNotEqual(depix_hmm.transition_probabilities[depix_hmm.states.index(('b',)), depix_hmm.states.index(('b',))], 0)
        for s in depix_hmm.transition_probabilities.sum(axis=1):
            self.assertAlmostEqual(s, 1.0, places=3)

        # Assert Emission Probabilities
        self.assertEqual(depix_hmm.emission_probabilities.shape, (len(depix_hmm.states), len(depix_hmm.observations)))
        for s in depix_hmm.emission_probabilities.sum(axis=1):
            self.assertAlmostEqual(s, 1.0, places=3)

    def test_test_image(self):
        # Arrange
        img_path: Path = Path(__file__).parent.parent.parent / 'examples' / 'arial_50_blocksize-8' / 'pixelized_cropped.png'

        picture_parameters: PictureParameters = PictureParameters(
            pattern=r'\d{9}',
            font=ImageFont.truetype(str(DemoFontPaths.arial), 50),
            block_size=8,
            window_size=4
        )

        training_parameters: TrainingParameters = TrainingParameters(
            n_img_train=100,
            n_img_test=1,
            n_clusters=150
        )
        depix_hmm: DepixHMM = DepixHMM(picture_parameters, training_parameters)
        depix_hmm.train()

        # Act
        with Image.open(img_path) as img:
            reconstructed_string: str = depix_hmm.test_image(img)

        # Assert
        self.assertIsInstance(reconstructed_string, str)
