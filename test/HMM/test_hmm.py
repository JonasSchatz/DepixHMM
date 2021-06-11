from typing import List
from unittest import TestCase

import numpy as np

from text_depixelizer.HMM.hmm import HMM


class TestHmm(TestCase):

    def create_random_hmm(self, observations, possible_states, possible_observations) -> HMM:
        """
        Returns a HMM object with random probabilities
        """

        n_possible_states: int = len(possible_states)
        n_possible_observations: int = len(possible_observations)

        starting_probabilities: np.ndarray = np.random.rand(n_possible_states)
        starting_probabilities = starting_probabilities / sum(starting_probabilities)

        transition_probabilities: np.ndarray = np.random.rand(n_possible_states, n_possible_states)
        transition_probabilities_normalized = transition_probabilities / np.sum(transition_probabilities, axis=1)[:, np.newaxis]

        emission_probabilities: np.ndarray = np.random.rand(n_possible_states, n_possible_observations)
        emission_probabilities_normalized = emission_probabilities / np.sum(emission_probabilities, axis=1)[:, np.newaxis]

        hmm: HMM = HMM(
            observations=observations,
            states=possible_states,
            starting_probabilities=starting_probabilities,
            transition_probabilities=transition_probabilities_normalized,
            emission_probabilities=emission_probabilities_normalized
        )

        return hmm

    def test_viterbi(self):
        # Arrange
        hmm: HMM = HMM(
            observations=[0, 1, 2],
            states=[('A', 'b'), ('b', )],
            starting_probabilities=np.array([0.7, 0.3]),
            transition_probabilities=np.array([[0.9, 0.1], [0.1, 0.9]]),
            emission_probabilities=np.array([[0.1, 0.4, 0.5], [0.3, 0.7, 0.0]])
        )

        sequence: List[int] = [2, 2, 2, 2, 2, 2, 2]

        # Act
        result = hmm.viterbi(sequence)

        # Assert
        self.assertEqual(len(result), len(sequence))
        self.assertTrue(all([r in hmm.states for r in result]))

    def test_viterbi_fail_for_numerical_underflow(self):
        """
        When the observation sequence gets too long, the regular viterbi will fail due to numerical underflow
        """

        # Parameters
        n_possible_observations: int = 100
        n_possible_states: int = 25
        observation_length: int = 10000

        # Arrange
        possible_observations: List[int] = list(range(n_possible_observations))
        possible_states: List[int] = list(range(n_possible_states))

        observations: List[int] = np.random.choice(possible_observations, size=observation_length)
        hmm: HMM = self.create_random_hmm(observations, possible_states, possible_observations)

        # Act
        result_viterbi = hmm.viterbi(observations)
        result_log_viterbi: List[int] = hmm.log_viterbi(observations)

        # Assert
        self.assertNotEqual(result_viterbi, result_log_viterbi)

    def test_compare_viterbi_and_log(self):
        """
        Regular viterbi and log-viterbi should return the same values (for shorter sequences)
        """
        np.random.seed(0)

        # Set parameters
        iterations: int = 50
        n_possible_observations: int = 100
        n_possible_states: int = 25
        max_observation_length: int = 100

        # Arrange (1)
        possible_observations: List[int] = list(range(n_possible_observations))
        possible_states: List[int] = list(range(n_possible_states))

        for i in range(iterations):
            # Arrange (2)
            observation_length: int = np.random.randint(1, max_observation_length)
            observations: List[int] = np.random.choice(possible_observations, size=observation_length)
            hmm: HMM = self.create_random_hmm(observations, possible_states, possible_observations)

            # Act
            result_viterbi = hmm.viterbi(observations)
            result_log_viterbi: List[int] = hmm.log_viterbi(observations)

            # Assert
            self.assertListEqual(result_viterbi, result_log_viterbi)
