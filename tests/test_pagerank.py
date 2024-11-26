import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import networkx as nx
from bioneuralnet.clustering.pagerank import PageRankClustering

class TestPageRankClustering(unittest.TestCase):

    @patch('pandas.read_excel')
    @patch('networkx.read_edgelist')
    def test_load_data(self, mock_read_edgelist, mock_read_excel):
        # Mock the read_edgelist function
        mock_G = nx.Graph()
        mock_G.add_edge('1', '2')
        mock_read_edgelist.return_value = mock_G

        # Mock the read_excel function
        mock_B = pd.DataFrame({'Gene1': [0.1, 0.2], 'Gene2': [0.3, 0.4]})
        mock_Y = pd.DataFrame({'Phenotype': [1, 0]})
        mock_read_excel.side_effect = [mock_B, mock_Y]

        pagerank_cluster = PageRankClustering(
            graph_file='input/graph.edgelist',
            omics_data_file='input/X.xlsx',
            phenotype_data_file='input/Y.xlsx'
        )

        pagerank_cluster.load_data()

        # Assertions
        self.assertEqual(pagerank_cluster.G, mock_G)
        self.assertTrue(pagerank_cluster.B.equals(mock_B))
        self.assertTrue((pagerank_cluster.Y == mock_Y.iloc[:,0]).all())

    @patch('networkx.pagerank')
    @patch('bioneuralnet.clustering.pagerank.PageRankClustering.load_data')
    @patch('bioneuralnet.clustering.pagerank.PageRankClustering.save_results')
    def test_run_pagerank_clustering(self, mock_save_results, mock_load_data, mock_pagerank):
        # Mock the load_data method
        mock_load_data.return_value = None

        # Mock the PageRankClustering instance variables
        pagerank_cluster = PageRankClustering(
            graph_file='input/graph.edgelist',
            omics_data_file='input/X.xlsx',
            phenotype_data_file='input/Y.xlsx'
        )
        pagerank_cluster.G = nx.Graph()
        pagerank_cluster.G.add_edge('1', '2')
        pagerank_cluster.B = pd.DataFrame({'Gene1': [0.1, 0.2], 'Gene2': [0.3, 0.4]})
        pagerank_cluster.Y = pd.Series([1, 0])

        # Mock the pagerank function
        mock_pagerank.return_value = {'1': 0.6, '2': 0.4}
        mock_pagerank.side_effect = lambda *args, **kwargs: {'1': 0.6, '2': 0.4}

        # Mock methods called within run_pagerank_clustering
        pagerank_cluster.generate_weighted_personalization = MagicMock(return_value={'1': 0.5, '2': 0.5})
        pagerank_cluster.sweep_cut = MagicMock(return_value=(['1', '2'], 2, 0.1, 0.8, 0.2, '0.8 (0.05)'))

        # Run the method
        results = pagerank_cluster.run_pagerank_clustering(seed_nodes=[1,2])

        # Assertions
        mock_pagerank.assert_called_once()
        pagerank_cluster.generate_weighted_personalization.assert_called_once_with([1,2])
        pagerank_cluster.sweep_cut.assert_called_once()
        mock_save_results.assert_called_once()
        self.assertIn('cluster_nodes', results)
        self.assertEqual(results['cluster_size'], 2)

if __name__ == '__main__':
    unittest.main()
