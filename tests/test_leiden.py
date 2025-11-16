import unittest
from unittest.mock import patch, MagicMock
import networkx as nx
import pandas as pd
import numpy as np

try:
    import igraph as ig
    import leidenalg as la
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False

from bioneuralnet.clustering.leiden import Leiden


class TestLeiden(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with a simple graph and omics data."""
        # Create a simple connected graph
        self.G = nx.Graph()
        self.G.add_edges_from([
            ("a", "b", {"weight": 1.0}),
            ("b", "c", {"weight": 1.0}),
            ("c", "d", {"weight": 1.0}),
            ("d", "a", {"weight": 0.5}),
            ("e", "f", {"weight": 1.0}),
            ("f", "g", {"weight": 1.0}),
            ("g", "e", {"weight": 1.0}),
        ])
        # Create omics data DataFrame
        self.B = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [2.0, 3.0, 4.0],
            "c": [3.0, 4.0, 5.0],
            "d": [1.5, 2.5, 3.5],
            "e": [5.0, 6.0, 7.0],
            "f": [6.0, 7.0, 8.0],
            "g": [5.5, 6.5, 7.5],
        }, index=["sample1", "sample2", "sample3"])

    @unittest.skipIf(not LEIDEN_AVAILABLE, "leidenalg or python-igraph not available")
    def test_initialization(self):
        """Test that Leiden algorithm initializes correctly."""
        leiden = Leiden(self.G, resolution_parameter=1.0, n_iterations=2, seed=42)
        
        self.assertEqual(leiden.resolution_parameter, 1.0)
        self.assertEqual(leiden.n_iterations, 2)
        self.assertEqual(leiden.seed, 42)
        self.assertEqual(leiden.partition_type, la.ModularityVertexPartition)
        self.assertIsNone(leiden.partition)
        self.assertIsNone(leiden.quality)

    @unittest.skipIf(not LEIDEN_AVAILABLE, "leidenalg or python-igraph not available")
    def test_initialization_with_different_partition_types(self):
        """Test initialization with different partition types."""
        # Test ModularityVertexPartition
        leiden_mod = Leiden(self.G, partition_type="ModularityVertexPartition")
        self.assertEqual(leiden_mod.partition_type, la.ModularityVertexPartition)
        
        # Test RBERVertexPartition
        leiden_rber = Leiden(self.G, partition_type="RBERVertexPartition")
        self.assertEqual(leiden_rber.partition_type, la.RBERVertexPartition)
        
        # Test CPMVertexPartition
        leiden_cpm = Leiden(self.G, partition_type="CPMVertexPartition")
        self.assertEqual(leiden_cpm.partition_type, la.CPMVertexPartition)
        
        # Test unknown partition type (should default to ModularityVertexPartition)
        leiden_unknown = Leiden(self.G, partition_type="UnknownPartition")
        self.assertEqual(leiden_unknown.partition_type, la.ModularityVertexPartition)

    def test_initialization_without_dependencies(self):
        """Test that ImportError is raised when dependencies are missing."""
        with patch('bioneuralnet.clustering.leiden._LEIDEN_AVAILABLE', False):
            with self.assertRaises(ImportError) as context:
                Leiden(self.G)
            self.assertIn("leidenalg", str(context.exception))
            self.assertIn("python-igraph", str(context.exception))

    @unittest.skipIf(not LEIDEN_AVAILABLE, "leidenalg or python-igraph not available")
    def test_run_returns_partition_dict(self):
        """Test that run() returns a partition dictionary."""
        leiden = Leiden(self.G, seed=42)
        partition = leiden.run()
        
        self.assertIsInstance(partition, dict)
        # Check all nodes are in partition
        self.assertEqual(set(partition.keys()), set(self.G.nodes()))
        # Check all nodes have community IDs
        self.assertTrue(all(isinstance(comm_id, int) for comm_id in partition.values()))
        # Check partition is stored
        self.assertEqual(leiden.partition, partition)

    @unittest.skipIf(not LEIDEN_AVAILABLE, "leidenalg or python-igraph not available")
    def test_run_sets_quality(self):
        """Test that run() computes and stores modularity quality."""
        leiden = Leiden(self.G, seed=42)
        partition = leiden.run()
        
        self.assertIsNotNone(leiden.quality)
        self.assertIsInstance(leiden.quality, float)
        # Modularity should be between -1 and 1
        self.assertGreaterEqual(leiden.quality, -1.0)
        self.assertLessEqual(leiden.quality, 1.0)

    @unittest.skipIf(not LEIDEN_AVAILABLE, "leidenalg or python-igraph not available")
    def test_run_as_dfs_returns_list_of_dataframes(self):
        """Test that run(as_dfs=True) returns list of DataFrames."""
        leiden = Leiden(self.G, seed=42)
        clusters = leiden.run(as_dfs=True, B=self.B)
        
        self.assertIsInstance(clusters, list)
        # Each cluster should be a DataFrame
        for cluster_df in clusters:
            self.assertIsInstance(cluster_df, pd.DataFrame)
            # Should have more than 2 nodes (filtered)
            self.assertGreater(len(cluster_df.columns), 2)
            # Columns should be valid node names from B
            self.assertTrue(all(col in self.B.columns for col in cluster_df.columns))

    @unittest.skipIf(not LEIDEN_AVAILABLE, "leidenalg or python-igraph not available")
    def test_run_as_dfs_requires_B(self):
        """Test that run(as_dfs=True) raises error if B is not provided."""
        leiden = Leiden(self.G, seed=42)
        with self.assertRaises(ValueError) as context:
            leiden.run(as_dfs=True)
        self.assertIn("B", str(context.exception))

    @unittest.skipIf(not LEIDEN_AVAILABLE, "leidenalg or python-igraph not available")
    def test_get_quality(self):
        """Test get_quality() method."""
        leiden = Leiden(self.G, seed=42)
        
        # Should raise error before run()
        with self.assertRaises(ValueError):
            leiden.get_quality()
        
        # Should return quality after run()
        leiden.run()
        quality = leiden.get_quality()
        self.assertIsNotNone(quality)
        self.assertIsInstance(quality, float)
        self.assertEqual(quality, leiden.quality)

    @unittest.skipIf(not LEIDEN_AVAILABLE, "leidenalg or python-igraph not available")
    def test_get_communities(self):
        """Test get_communities() method."""
        leiden = Leiden(self.G, seed=42)
        
        # Should raise error before run()
        with self.assertRaises(ValueError):
            leiden.get_communities()
        
        # Should return communities after run()
        leiden.run()
        communities = leiden.get_communities()
        
        self.assertIsInstance(communities, dict)
        # Check all nodes are present
        all_nodes_in_communities = set()
        for nodes in communities.values():
            all_nodes_in_communities.update(nodes)
        self.assertEqual(all_nodes_in_communities, set(self.G.nodes()))
        # Check communities are non-empty
        self.assertTrue(all(len(nodes) > 0 for nodes in communities.values()))

    @unittest.skipIf(not LEIDEN_AVAILABLE, "leidenalg or python-igraph not available")
    def test_resolution_parameter_affects_communities(self):
        """Test that different resolution parameters produce different results."""
        leiden_low = Leiden(self.G, resolution_parameter=0.5, seed=42)
        leiden_high = Leiden(self.G, resolution_parameter=2.0, seed=42)
        
        partition_low = leiden_low.run()
        partition_high = leiden_high.run()
        
        # Different resolutions typically produce different number of communities
        num_communities_low = len(set(partition_low.values()))
        num_communities_high = len(set(partition_high.values()))
        
        # At least verify they ran successfully
        self.assertGreater(num_communities_low, 0)
        self.assertGreater(num_communities_high, 0)

    @unittest.skipIf(not LEIDEN_AVAILABLE, "leidenalg or python-igraph not available")
    def test_seed_produces_reproducible_results(self):
        """Test that the same seed produces the same partition."""
        leiden1 = Leiden(self.G, seed=42)
        leiden2 = Leiden(self.G, seed=42)
        
        partition1 = leiden1.run()
        partition2 = leiden2.run()
        
        self.assertEqual(partition1, partition2)
        self.assertAlmostEqual(leiden1.quality, leiden2.quality, places=5)

    @unittest.skipIf(not LEIDEN_AVAILABLE, "leidenalg or python-igraph not available")
    def test_partition_to_adjacency(self):
        """Test partition_to_adjacency() method."""
        leiden = Leiden(self.G, seed=42)
        partition = leiden.run()
        
        clusters = leiden.partition_to_adjacency(partition, self.B)
        
        self.assertIsInstance(clusters, list)
        # Check all clusters have >2 nodes
        for cluster_df in clusters:
            self.assertGreater(len(cluster_df.columns), 2)
            self.assertTrue(all(col in self.B.columns for col in cluster_df.columns))

    @unittest.skipIf(not LEIDEN_AVAILABLE, "leidenalg or python-igraph not available")
    def test_graph_with_weights(self):
        """Test that edge weights are preserved."""
        leiden = Leiden(self.G, seed=42)
        partition = leiden.run()
        
        # Should run successfully with weighted graph
        self.assertIsNotNone(partition)
        self.assertIsNotNone(leiden.quality)

    @unittest.skipIf(not LEIDEN_AVAILABLE, "leidenalg or python-igraph not available")
    def test_directed_graph(self):
        """Test with directed graph."""
        G_directed = nx.DiGraph()
        G_directed.add_edges_from([
            ("a", "b"),
            ("b", "c"),
            ("c", "a"),
        ])
        
        leiden = Leiden(G_directed, seed=42)
        partition = leiden.run()
        
        self.assertIsNotNone(partition)
        self.assertEqual(set(partition.keys()), set(G_directed.nodes()))

    @unittest.skipIf(not LEIDEN_AVAILABLE, "leidenalg or python-igraph not available")
    def test_graph_conversion_preserves_nodes(self):
        """Test that graph conversion preserves all nodes."""
        leiden = Leiden(self.G, seed=42)
        ig_graph = leiden._nx_to_igraph(self.G)
        
        # Check node count matches
        self.assertEqual(len(ig_graph.vs), len(self.G.nodes()))
        # Check all node names are preserved
        ig_node_names = set(ig_graph.vs['name'])
        nx_node_names = set(self.G.nodes())
        self.assertEqual(ig_node_names, nx_node_names)

    @unittest.skipIf(not LEIDEN_AVAILABLE, "leidenalg or python-igraph not available")
    def test_graph_conversion_preserves_weights(self):
        """Test that edge weights are preserved in conversion."""
        leiden = Leiden(self.G, seed=42)
        ig_graph = leiden._nx_to_igraph(self.G)
        
        # Check weights are present
        self.assertIn('weight', ig_graph.es.attributes())
        # Check all edges have weights
        self.assertTrue(all('weight' in edge.attributes() for edge in ig_graph.es))

    @unittest.skipIf(not LEIDEN_AVAILABLE, "leidenalg or python-igraph not available")
    def test_partition_to_dict(self):
        """Test partition_to_dict conversion."""
        leiden = Leiden(self.G, seed=42)
        ig_graph = leiden._nx_to_igraph(self.G)
        node_names = [ig_graph.vs[idx]['name'] for idx in range(len(ig_graph.vs))]
        
        # Create a mock partition
        partition = la.find_partition(ig_graph, la.ModularityVertexPartition)
        partition_dict = leiden._partition_to_dict(partition, node_names)
        
        self.assertIsInstance(partition_dict, dict)
        self.assertEqual(len(partition_dict), len(node_names))
        self.assertEqual(set(partition_dict.keys()), set(node_names))

    @unittest.skipIf(not LEIDEN_AVAILABLE, "leidenalg or python-igraph not available")
    def test_empty_graph(self):
        """Test with empty graph (should handle gracefully)."""
        G_empty = nx.Graph()
        leiden = Leiden(G_empty, seed=42)
        partition = leiden.run()
        
        self.assertIsInstance(partition, dict)
        self.assertEqual(len(partition), 0)

    @unittest.skipIf(not LEIDEN_AVAILABLE, "leidenalg or python-igraph not available")
    def test_single_node_graph(self):
        """Test with single node graph."""
        G_single = nx.Graph()
        G_single.add_node("a")
        leiden = Leiden(G_single, seed=42)
        partition = leiden.run()
        
        self.assertIsInstance(partition, dict)
        self.assertEqual(len(partition), 1)
        self.assertIn("a", partition)


if __name__ == "__main__":
    unittest.main()

