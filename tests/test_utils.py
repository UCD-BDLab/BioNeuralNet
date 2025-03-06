import unittest
import pandas as pd
from bioneuralnet.utils.variance import (
    remove_variance,
    remove_fraction,
    network_remove_low_variance,
    network_remove_high_zero_fraction,
    network_filter,
    omics_data_filter,
)

class TestVarianceFunctions(unittest.TestCase):
    def test_remove_variance(self):
        df = pd.DataFrame({"A": [1, 1, 1, 1], "B": [1, 2, 3, 4], "C": [2, 2, 2, 2]})
        filtered = remove_variance(df, variance_threshold=0.0)
        self.assertNotIn("A", filtered.columns)
        self.assertNotIn("C", filtered.columns)
        self.assertIn("B", filtered.columns)

    def test_remove_fraction(self):
        df = pd.DataFrame({"A": [0, 0, 0, 0], "B": [1, 0, 1, 1], "C": [0, 1, 0, 1]})
        filtered = remove_fraction(df, zero_frac_threshold=0.75)
        self.assertNotIn("A", filtered.columns)
        self.assertIn("B", filtered.columns)

    def test_network_remove_low_variance(self):
        df = pd.DataFrame(
            {"A": [1, 1, 1], "B": [1, 2, 3], "C": [2, 3, 4]},
            index=["A", "B", "C"],
        )
        filtered = network_remove_low_variance(df, threshold=0.1)
        self.assertNotIn("A", filtered.columns)

    def test_network_remove_high_zero_fraction(self):
        df = pd.DataFrame(
            {"A": [0, 0, 0], "B": [1, 2, 3], "C": [2, 3, 4]},
            index=["A", "B", "C"],
        )
        filtered = network_remove_high_zero_fraction(df, threshold=0.66)
        self.assertNotIn("A", filtered.columns)
        self.assertIn("B", filtered.columns)
        self.assertIn("C", filtered.columns)

    def test_network_filter_variance(self):
        df = pd.DataFrame(
            {"A": [1, 1, 1], "B": [1, 2, 3], "C": [2, 3, 4]},
            index=["A", "B", "C"],
        )
        filtered = network_filter(df, threshold=0.1, filter_type="variance")
        self.assertNotIn("A", filtered.columns)

    def test_network_filter_zero_fraction(self):
        df = pd.DataFrame(
            {"A": [0, 0, 0], "B": [1, 2, 3], "C": [2, 3, 4]},
            index=["A", "B", "C"],
        )
        filtered = network_filter(df, threshold=0.66, filter_type="zero_fraction")
        self.assertNotIn("A", filtered.columns)
        self.assertListEqual(list(filtered.columns), ["B", "C"])

    def test_omics_data_filter(self):
        df = pd.DataFrame({"A": [0, 0, 0, 0], "B": [1, 2, 3, 4], "C": [0, 1, 0, 1]})
        filtered = omics_data_filter(df, variance_threshold=0.1, zero_frac_threshold=0.75)
        self.assertNotIn("A", filtered.columns)

if __name__ == "__main__":
    unittest.main()
