from unittest import TestCase

from test.utils import demo_picture_parameters
from text_depixelizer.HMM.clusterer import KmeansClusterer
from text_depixelizer.training_pipeline.training_pipeline import create_training_data


class TestKmeansClusterer(TestCase):

    def test_kmeans_fit(self):
        # Arrange
        _, _, _, windows = create_training_data(n_img=1, picture_parameters=demo_picture_parameters)

        # Act
        kmeans_clusterer: KmeansClusterer = KmeansClusterer(windows=windows[0], k=5)

        # Assert
        self.assertEqual(kmeans_clusterer.kmeans.n_clusters, 5)
