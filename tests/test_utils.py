# import unittest
# import pandas as pd
# from bioneuralnet.utils.data_summary import (
#     network_remove_low_variance,
#     network_remove_high_zero_fraction,
#     network_filter,
# )

# class TestVarianceFunctions(unittest.TestCase):

#     def test_network_remove_low_variance(self):
#         df = pd.DataFrame(
#             {"A": [1, 1, 1], "B": [1, 2, 3], "C": [2, 3, 4]},
#             index=["A", "B", "C"],
#         )
#         filtered = network_remove_low_variance(df, threshold=0.1)
#         self.assertNotIn("A", filtered.columns)

#     def test_network_remove_high_zero_fraction(self):
#         df = pd.DataFrame(
#             {"A": [0, 0, 0], "B": [1, 2, 3], "C": [2, 3, 4]},
#             index=["A", "B", "C"],
#         )
#         filtered = network_remove_high_zero_fraction(df, threshold=0.66)
#         self.assertNotIn("A", filtered.columns)
#         self.assertIn("B", filtered.columns)
#         self.assertIn("C", filtered.columns)

#     def test_network_filter_variance(self):
#         df = pd.DataFrame(
#             {"A": [1, 1, 1], "B": [1, 2, 3], "C": [2, 3, 4]},
#             index=["A", "B", "C"],
#         )
#         filtered = network_filter(df, threshold=0.1, filter_type="variance")
#         self.assertNotIn("A", filtered.columns)

#     def test_network_filter_zero_fraction(self):
#         df = pd.DataFrame(
#             {"A": [0, 0, 0], "B": [1, 2, 3], "C": [2, 3, 4]},
#             index=["A", "B", "C"],
#         )
#         filtered = network_filter(df, threshold=0.66, filter_type="zero_fraction")
#         self.assertNotIn("A", filtered.columns)
#         self.assertListEqual(list(filtered.columns), ["B", "C"])


# if __name__ == "__main__":
#     unittest.main()
