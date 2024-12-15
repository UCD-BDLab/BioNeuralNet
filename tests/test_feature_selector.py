import unittest
import pandas as pd
import os
from bioneuralnet.analysis.feature_selector import FeatureSelector

class TestFeatureSelector(unittest.TestCase):
    def setUp(self):
        self.enhanced_omics_data = pd.DataFrame({
            'gene1': [1, 2, 3, 4, 5],
            'gene2': [5, 4, 3, 2, 1],
            'gene3': [2, 3, 4, 5, 6],
            'protein1': [3, 4, 5, 6, 7],
            'metabolite1': [4, 5, 6, 7, 8]
        }, index=['sample1', 'sample2', 'sample3', 'sample4', 'sample5'])

        self.phenotype_data = pd.Series([0, 1, 0, 1, 0], index=['sample1', 'sample2', 'sample3', 'sample4', 'sample5'], name='Asthma')

        self.feature_selector = FeatureSelector(
            enhanced_omics_data=self.enhanced_omics_data,
            phenotype_data=self.phenotype_data,
            num_features=2,
            selection_method='correlation',
        )

    def tearDown(self):
        if os.path.exists(self.feature_selector.output_dir):
            for f in os.listdir(self.feature_selector.output_dir):
                os.remove(os.path.join(self.feature_selector.output_dir, f))
            os.rmdir(self.feature_selector.output_dir)

    def test_run_feature_selection_correlation(self):
        selected_features = self.feature_selector.run_feature_selection()
        self.assertEqual(selected_features.shape[1], 2)
        expected_features = ['gene1', 'gene2']
        self.assertListEqual(list(selected_features.columns), expected_features)

    def test_run_feature_selection_lasso(self):
        self.feature_selector.selection_method = 'lasso'
        selected_features = self.feature_selector.run_feature_selection()
        self.assertEqual(selected_features.shape[1], 2)
        # Based on mock data, gene1 and gene3 should be selected
        expected_features = ['gene1', 'gene3']
        self.assertListEqual(list(selected_features.columns), expected_features)

    def test_run_feature_selection_random_forest(self):
        self.feature_selector.selection_method = 'random_forest'
        selected_features = self.feature_selector.run_feature_selection()
        self.assertEqual(selected_features.shape[1], 2)
        expected_features = ['gene2', 'gene3']
        self.assertListEqual(list(selected_features.columns), expected_features)

    def test_invalid_selection_method(self):
        self.feature_selector.selection_method = 'invalid_method'
        with self.assertRaises(ValueError):
            self.feature_selector.run_feature_selection()

    def test_save_selected_features(self):
        selected_features = self.feature_selector.run_feature_selection()
        expected_file = os.path.join(self.feature_selector.output_dir, "selected_genetic_features.csv")
        self.assertTrue(os.path.exists(expected_file))
        saved_features = pd.read_csv(expected_file, index_col=0)
        pd.testing.assert_frame_equal(selected_features, saved_features)

if __name__ == '__main__':
    unittest.main()
