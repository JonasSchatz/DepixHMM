import logging
from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Any

import numpy as np


class HmmAttributeException(Exception):
    pass


@dataclass
class HMM:
    observations: List[Any]
    states: List[Any]
    starting_probabilities: np.ndarray
    transition_probabilities: np.ndarray
    emission_probabilities: np.ndarray

    @cached_property
    def log_starting_probabilities(self) -> np.ndarray:
        return np.log(self.starting_probabilities)

    @cached_property
    def log_transition_probabilities(self) -> np.ndarray:
        return np.log(self.transition_probabilities)

    @cached_property
    def log_emission_probabilities(self) -> np.ndarray:
        return np.log(self.emission_probabilities)

    def validate_attributes(self) -> None:
        if len(self.starting_probabilities) != len(self.states):
            raise HmmAttributeException('Starting probabilities must have one entry for each state!')

        if self.transition_probabilities.shape != (len(self.states), len(self.states)):
            raise HmmAttributeException('Transition probabilities must have shape (n_states, n_shapes)')

        if not all(np.sum(self.transition_probabilities, axis=1) == 1):
            logging.warning('Careful, transition probabilities not properly normalized')

        if self.emission_probabilities.shape != (len(self.states), len(self.observations)):
            raise HmmAttributeException('Emission probabilities must have shape (n_states, n_observations)')

        if not all(np.sum(self.emission_probabilities, axis=1) == 1):
            logging.warning('Careful, transition probabilities not properly normalized')

    def viterbi(self, sequence: List[Any]):
        # Initialize tables
        v: np.ndarray[float, float] = np.zeros((len(self.states), len(sequence)))
        pointers: np.ndarray[Optional[float], Optional[float]] = np.empty(v.shape)

        for i, observation in enumerate(sequence):
            if i==0:
                v[:, i] = self.starting_probabilities * self.emission_probabilities[:, observation]
                pointers[:, i] = 0

            else:
                v[:, i] = np.max(v[:, i-1] * self.transition_probabilities.T * self.emission_probabilities[np.newaxis, :, sequence[i]].T, 1)
                pointers[:, i] = np.argmax(v[:, i-1] * self.transition_probabilities.T, 1)

        x = np.empty(len(sequence), 'B')
        x[-1] = np.argmax(v[:, len(sequence)-1])
        for i in reversed(range(1, len(sequence))):
            x[i-1] = pointers[x[i], i]

        return [self.states[i] for i in x]

    def log_viterbi(self, sequence: List[Any]):
        # Initialize tables
        v: np.ndarray[float, float] = np.zeros((len(self.states), len(sequence)))
        pointers: np.ndarray[Optional[float], Optional[float]] = np.empty(v.shape)

        for i, observation in enumerate(sequence):
            if i==0:
                v[:, i] = self.log_starting_probabilities + self.log_emission_probabilities[:, observation]
                pointers[:, i] = 0

            else:
                v[:, i] = np.max(v[:, i-1] + self.log_transition_probabilities.T + self.log_emission_probabilities[np.newaxis, :, sequence[i]].T, 1)
                pointers[:, i] = np.argmax(v[:, i-1] + self.log_transition_probabilities.T, 1)

        x = np.empty(len(sequence), 'B')
        x[-1] = np.argmax(v[:, len(sequence)-1])
        for i in reversed(range(1, len(sequence))):
            x[i-1] = pointers[x[i], i]

        return [self.states[i] for i in x]

