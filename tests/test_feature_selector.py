import os
import unittest
import pandas as pd
from bioneuralnet.external_tools import FeatureSelector


class TestFeatureSelector(unittest.TestCase):

    def setUp(self):
        self.enhanced_omics_data = pd.DataFrame(
            {
                "feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
                "feature2": [1.2, 1.3, 1.1, 1.4, 1.5],
                "feature3": [2.1, 2.2, 2.3, 2.4, 2.5],
                "feature4": [3.1, 3.2, 3.3, 3.4, 3.5],
                "feature5": [4.1, 4.2, 4.3, 4.4, 4.5],
                "feature6": [5.1, 5.2, 5.3, 5.4, 5.5],
                "feature7": [6.1, 6.2, 6.3, 6.4, 6.5],
                "feature8": [7.1, 7.2, 7.3, 7.4, 7.5],
                "feature9": [8.1, 8.2, 8.3, 8.4, 8.5],
                "feature10": [9.1, 9.2, 9.3, 9.4, 9.5],
                "feature11": [10.1, 10.2, 10.3, 10.4, 10.5],
                "feature12": [11.1, 11.2, 11.3, 11.4, 11.5],
                "feature13": [12.1, 12.2, 12.3, 12.4, 12.5],
                "feature14": [13.1, 13.2, 13.3, 13.4, 13.5],
                "feature15": [14.1, 14.2, 14.3, 14.4, 14.5],
            }
        )

        self.phenotype_data = pd.Series([0, 1, 0, 1, 0], name="Phenotype")

    def test_initialization(self):
        """
        Test the initialization of the FeatureSelector class.
        """
        fs = FeatureSelector(
            enhanced_omics_data=self.enhanced_omics_data,
            phenotype_data=self.phenotype_data,
            num_features=5,
            selection_method="correlation",
        )
        self.assertEqual(fs.num_features, 5)
        self.assertEqual(fs.selection_method, "correlation")
        self.assertIsInstance(fs.enhanced_omics_data, pd.DataFrame)
        self.assertIsInstance(fs.phenotype_data, pd.Series)
        self.assertTrue(hasattr(fs, "logger"))

    def test_correlation_selection(self):
        """
        Test correlation-based feature selection.
        """
        fs = FeatureSelector(
            enhanced_omics_data=self.enhanced_omics_data,
            phenotype_data=self.phenotype_data,
            num_features=5,
            selection_method="correlation",
        )
        selected = fs.perform_feature_selection()
        self.assertEqual(selected.shape[1], 5)

    def test_lasso_selection(self):
        """
        Test LASSO-based feature selection.
        """
        fs = FeatureSelector(
            enhanced_omics_data=self.enhanced_omics_data,
            phenotype_data=self.phenotype_data,
            num_features=5,
            selection_method="lasso",
        )
        selected = fs.perform_feature_selection()
        self.assertEqual(selected.shape[1], 5)

    def test_random_forest_selection(self):
        """
        Test Random Forest-based feature selection.
        """
        fs = FeatureSelector(
            enhanced_omics_data=self.enhanced_omics_data,
            phenotype_data=self.phenotype_data,
            num_features=5,
            selection_method="random_forest",
        )
        selected = fs.perform_feature_selection()
        self.assertEqual(selected.shape[1], 5)

    def test_invalid_selection_method(self):
        """
        Test that an invalid selection method raises a ValueError.
        """
        fs = FeatureSelector(
            enhanced_omics_data=self.enhanced_omics_data,
            phenotype_data=self.phenotype_data,
            num_features=5,
            selection_method="invalid_method",
        )
        with self.assertRaises(ValueError):
            fs.perform_feature_selection()

    def test_run_feature_selection(self):
        """
        Test the full feature selection run.
        """
        fs = FeatureSelector(
            enhanced_omics_data=self.enhanced_omics_data,
            phenotype_data=self.phenotype_data,
            num_features=5,
            selection_method="correlation",
        )
        selected = fs.run_feature_selection()
        self.assertEqual(selected.shape[1], 5)


if __name__ == "__main__":
    unittest.main()
