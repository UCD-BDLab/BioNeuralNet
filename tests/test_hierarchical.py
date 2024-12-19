import unittest
import pandas as pd
from bioneuralnet.clustering import HierarchicalClustering

class TestHierarchicalClustering(unittest.TestCase):

    def setUp(self):
        # Sample adjacency matrix
        self.adjacency_matrix = pd.DataFrame({
            'GeneA': [1.0, 0.8, 0.3],
            'GeneB': [0.8, 1.0, 0.4],
            'GeneC': [0.3, 0.4, 1.0]
        }, index=['GeneA', 'GeneB', 'GeneC'])

    def test_clustering_output(self):
        hc = HierarchicalClustering(adjacency_matrix=self.adjacency_matrix, n_clusters=2, linkage='ward', affinity='euclidean')
        results = hc.run()
        
        # Check if cluster_labels is a DataFrame
        self.assertIsInstance(results['cluster_labels'], pd.DataFrame)
        
        # Check if silhouette_score is a float or None
        self.assertTrue(isinstance(results['silhouette_score'], float) or results['silhouette_score'] is None)
        
        # Check the number of unique clusters
        unique_clusters = results['cluster_labels']['cluster'].nunique()
        self.assertEqual(unique_clusters, 2)

    def test_without_running_clustering(self):
        hc = HierarchicalClustering(adjacency_matrix=self.adjacency_matrix)
        with self.assertRaises(ValueError):
            hc.get_results()

    def test_invalid_affinity_with_ward_linkage(self):
        hc = HierarchicalClustering(adjacency_matrix=self.adjacency_matrix, linkage='ward', affinity='cosine')
        with self.assertRaises(ValueError):
            hc.run()

if __name__ == '__main__':
    unittest.main()
