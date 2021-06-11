from abc import ABC, abstractmethod
from typing import List

import numpy as np
from sklearn.cluster import KMeans

from text_depixelizer.training_pipeline.windows import Window


class Clusterer(ABC):
    centroids: List[np.ndarray]

    @abstractmethod
    def map_windows_to_cluster(self, windows: List[Window]) -> List[Window]:
        pass

    @abstractmethod
    def map_values_to_cluster(self, values: List[np.array]) -> List[int]:
        pass

class KmeansClusterer(Clusterer):
    kmeans: KMeans

    def __init__(self, windows: List[Window], k: int):
        X = np.array([window.values for window in windows])
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        self.kmeans = kmeans

    def map_windows_to_cluster(self, windows: List[Window]) -> List[Window]:
        k_values: List[int] = self.map_values_to_cluster([window.values for window in windows])
        for window, k_value in zip(windows, k_values):
            window.k = k_value
        return windows

    def map_values_to_cluster(self, values: List[np.array]) -> List[int]:
        k_values: List[int] = self.kmeans.predict(np.array(values))
        return k_values
