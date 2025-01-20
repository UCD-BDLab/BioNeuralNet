import unittest
import pandas as pd
from bioneuralnet.external_tools import HierarchicalClustering


class TestHierarchicalClustering(unittest.TestCase):

    def setUp(self):
        self.adjacency_matrix = pd.DataFrame(
            {
                "GeneA": [1.0, 0.5, 0.2],
                "GeneB": [0.3, 1.0, 0.7],
                "GeneC": [0.4, 0.6, 1.0],
            },
            index=["GeneA", "GeneB", "GeneC"],
        )

    def test_clustering_output(self):
        hc = HierarchicalClustering(
            adjacency_matrix=self.adjacency_matrix,
            n_clusters=2,
            linkage="ward",
            affinity="euclidean",
            scale_data=True,
        )
        results = hc.run()
        self.assertIsInstance(results["cluster_labels"], pd.DataFrame)
        self.assertTrue(
            isinstance(results["silhouette_score"], float)
            or results["silhouette_score"] is None
        )
        unique_clusters = results["cluster_labels"]["cluster"].nunique()
        self.assertEqual(unique_clusters, 2)

    def test_without_running_clustering(self):
        hc = HierarchicalClustering(adjacency_matrix=self.adjacency_matrix)
        with self.assertRaises(ValueError):
            hc.get_results()

    def test_invalid_metric_with_ward_linkage(self):
        with self.assertRaises(
            ValueError,
            msg="The 'ward' linkage method only supports the 'euclidean' metric.",
        ):
            HierarchicalClustering(
                adjacency_matrix=self.adjacency_matrix,
                linkage="ward",
                affinity="cosine",
            ).run()

    def test_invalid_metric_without_ward_linkage(self):
        with self.assertRaises(
            ValueError, msg="Unsupported affinity metric: invalid_metric"
        ):
            HierarchicalClustering(
                adjacency_matrix=self.adjacency_matrix,
                linkage="complete",
                affinity="invalid_metric",
            ).run()

    def test_scaling_data(self):
        hc = HierarchicalClustering(
            adjacency_matrix=self.adjacency_matrix,
            n_clusters=2,
            linkage="complete",
            affinity="euclidean",
            scale_data=True,
        )
        hc.run()
        self.assertIsNotNone(hc.scaled_feature_matrix)

    def test_no_scaling_data(self):
        hc = HierarchicalClustering(
            adjacency_matrix=self.adjacency_matrix,
            n_clusters=2,
            linkage="complete",
            affinity="euclidean",
            scale_data=False,
        )
        hc.run()
        self.assertIsNotNone(hc.scaled_feature_matrix)
        pd.testing.assert_frame_equal(hc.scaled_feature_matrix, self.adjacency_matrix)


if __name__ == "__main__":
    unittest.main()
