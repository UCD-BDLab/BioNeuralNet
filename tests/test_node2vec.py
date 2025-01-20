# import os
# import unittest
# import pandas as pd
# from bioneuralnet.external_tools import node2vec


# class Testnode2vec(unittest.TestCase):

#     def setUp(self):
#         self.adjacency_matrix = pd.DataFrame(
#             {
#                 "GeneA": [1.0, 1.0, 0.0],
#                 "GeneB": [1.0, 1.0, 1.0],
#                 "GeneC": [0.0, 1.0, 1.0],
#             },
#             index=["GeneA", "GeneB", "GeneC"],
#         )

#     def test_embedding_output_default_parameters(self):
#         node2vec_emebdding = node2vec(adjacency_matrix=self.adjacency_matrix)
#         embeddings = node2vec_emebdding.run()

#         self.assertIsInstance(
#             embeddings,
#             pd.DataFrame,
#             f"Expected DataFrame, got {type(embeddings)} instead.",
#         )

#         self.assertIn(
#             "node",
#             embeddings.columns,
#             "'node' column is missing in the embeddings DataFrame.",
#         )

#         expected_columns = ["node"]
#         for i in range(128):
#             expected_columns.append(str(i))

#         all_columns_present = True
#         for col in expected_columns:
#             if col not in embeddings.columns:
#                 all_columns_present = False
#                 break

#         self.assertTrue(
#             all_columns_present,
#             f"Embeddings DataFrame is missing expected columns. Found: {embeddings.columns}",
#         )

#     def test_embedding_output_custom_parameters(self):
#         node2vec_emebdding = node2vec(
#             adjacency_matrix=self.adjacency_matrix,
#             embedding_dim=64,
#             walk_length=30,
#             num_walks=200,
#             window_size=10,
#             workers=2,
#             seed=123,
#             p=0.5,
#             q=2.0,
#             weight_key="weight",
#         )
#         embeddings = node2vec_emebdding.run()

#         self.assertIsInstance(
#             embeddings,
#             pd.DataFrame,
#             f"Expected DataFrame, got {type(embeddings)} instead.",
#         )

#         self.assertIn(
#             "node",
#             embeddings.columns,
#             "'node' column is missing in the embeddings DataFrame.",
#         )

#         expected_columns = ["node"]
#         for i in range(64):
#             expected_columns.append(str(i))

#         all_columns_present = True
#         for col in expected_columns:
#             if col not in embeddings.columns:
#                 all_columns_present = False
#                 break

#         self.assertTrue(
#             all_columns_present,
#             f"Embeddings DataFrame is missing expected columns. Found: {embeddings.columns}",
#         )

#     def test_embedding_with_custom_weight_key(self):
#         adjacency_matrix = pd.DataFrame(
#             {
#                 "GeneA": [1.0, 2.0, 0.0],
#                 "GeneB": [2.0, 1.0, 3.0],
#                 "GeneC": [0.0, 3.0, 1.0],
#             },
#             index=["GeneA", "GeneB", "GeneC"],
#         )

#         node2vec_emebdding = node2vec(
#             adjacency_matrix=adjacency_matrix,
#             weight_key="custom_weight",
#             embedding_dim=64,
#             walk_length=30,
#             num_walks=200,
#             window_size=10,
#             workers=2,
#             seed=123,
#             p=0.5,
#             q=2.0,
#         )
#         embeddings = node2vec_emebdding.run()

#         self.assertIsInstance(
#             embeddings,
#             pd.DataFrame,
#             f"Expected DataFrame, got {type(embeddings)} instead.",
#         )

#         self.assertIn(
#             "node",
#             embeddings.columns,
#             "'node' column is missing in the embeddings DataFrame.",
#         )

#         expected_columns = ["node"]
#         for i in range(64):
#             expected_columns.append(str(i))

#         all_columns_present = True
#         for col in expected_columns:
#             if col not in embeddings.columns:
#                 all_columns_present = False
#                 break

#         self.assertTrue(
#             all_columns_present,
#             f"Embeddings DataFrame is missing expected columns. Found: {embeddings.columns}",
#         )

#     def test_get_embeddings_before_run(self):
#         node2vec_emebdding = node2vec(adjacency_matrix=self.adjacency_matrix)
#         with self.assertRaises(ValueError) as context:
#             node2vec_emebdding.get_embeddings()
#         self.assertIn("Embeddings have not been generated yet", str(context.exception))

#     def test_save_embeddings_before_run(self):
#         node2vec_emebdding = node2vec(adjacency_matrix=self.adjacency_matrix)
#         with self.assertRaises(ValueError) as context:
#             node2vec_emebdding.save_embeddings("embeddings.csv")
#         self.assertIn("Embeddings have not been generated yet", str(context.exception))

#     def test_save_embeddings_after_run(self):
#         node2vec_emebdding = node2vec(
#             adjacency_matrix=self.adjacency_matrix,
#             embedding_dim=64,
#             walk_length=30,
#             num_walks=200,
#             window_size=10,
#             workers=2,
#             seed=123,
#             p=0.5,
#             q=2.0,
#             weight_key="weight",
#         )
#         embeddings = node2vec_emebdding.run()
#         embeddings.save_embeddings("test_embeddings.csv")

#         self.assertTrue(
#             os.path.exists("test_embeddings.csv"), "Embeddings file was not created."
#         )

#         try:
#             os.remove("test_embeddings.csv")
#         except Exception as e:
#             self.fail(f"Failed to remove test embeddings file: {e}")


# if __name__ == "__main__":
#     unittest.main()
