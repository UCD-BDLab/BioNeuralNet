# import unittest
# from unittest.mock import patch, MagicMock
# import pandas as pd
# from bioneuralnet.network_embedding.node2vec import Node2VecEmbedding

# class TestNode2VecEmbedding(unittest.TestCase):

#     @patch('bioneuralnet.network_embedding.node2vec.Node2Vec')
#     @patch('bioneuralnet.network_embedding.node2vec.Node2VecEmbedding.load_graphs_from_directory')
#     def test_run_node2vec_embedding_success(self, mock_load_graphs, mock_node2vec_class):
#         # Mock the load_graphs_from_directory method
#         mock_graphs = {
#             'graph1': pd.DataFrame([[0,1],[1,0]], index=['node1','node2'], columns=['node1','node2'])
#         }
#         mock_load_graphs.return_value = mock_graphs

#         # Mock the Node2Vec model
#         mock_model = MagicMock()
#         mock_wv = MagicMock()
#         mock_wv.vectors = [[0.1, 0.2], [0.3, 0.4]]
#         mock_wv.index_to_key = ['node1', 'node2']
#         mock_model.fit.return_value = MagicMock(wv=mock_wv)
#         mock_node2vec_class.return_value = mock_model

#         # Initialize Node2VecEmbedding
#         node2vec_embedding = Node2VecEmbedding(
#             input_dir='input/graphs/',
#             embedding_dim=128,
#             walk_length=80,
#             num_walks=10,
#             window_size=10,
#             workers=4,
#             seed=42,
#             output_dir='test_output_dir'
#         )

#         # Run the embedding
#         embeddings = node2vec_embedding.run()

#         # Assertions
#         mock_load_graphs.assert_called_once()
#         mock_node2vec_class.assert_called_once()
#         self.assertIn('graph1', embeddings)
#         embeddings_df = embeddings['graph1']
#         self.assertEqual(len(embeddings_df), 2)
#         self.assertListEqual(embeddings_df['node'].tolist(), ['node1', 'node2'])
#         self.assertEqual(embeddings_df.shape[1], 1 + node2vec_embedding.embedding_dim)

#     @patch('bioneuralnet.network_embedding.node2vec.find_files')
#     @patch('pandas.read_csv')
#     def test_load_graphs_from_directory(self, mock_read_csv, mock_find_files):
#         # Mock the find_files function
#         mock_find_files.return_value = ['graph1.csv', 'graph2.csv']

#         # Mock the pandas.read_csv function
#         mock_adjacency_matrix = pd.DataFrame([[0,1],[1,0]], index=['node1','node2'], columns=['node1','node2'])
#         mock_read_csv.return_value = mock_adjacency_matrix

#         node2vec_embedding = Node2VecEmbedding(input_dir='input/graphs/')

#         graphs = node2vec_embedding.load_graphs_from_directory()

#         # Assertions
#         self.assertEqual(len(graphs), 2)
#         self.assertIn('graph1', graphs)
#         self.assertIn('graph2', graphs)
#         self.assertTrue(graphs['graph1'].equals(mock_adjacency_matrix))

# if __name__ == '__main__':
#     unittest.main()
