import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
pd_notebooks = project_root / "PD-Notebooks"
if str(pd_notebooks) not in sys.path:
    sys.path.insert(0, str(pd_notebooks))

from models.baseline.baseline_model import (
    BaselineModel,
    BaselineResults,
    train_baseline,
)


class TestPDBaseline(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_genes = 50
        n_samples = 20

        # Create expression data
        self.expression_df = pd.DataFrame(
            np.random.randn(n_genes, n_samples),
            index=[f"GENE_{i}" for i in range(n_genes)],
            columns=[f"SAMPLE_{i}" for i in range(n_samples)],
        )

        # Create sample metadata with PD/CC labels
        conditions = ["PD"] * 10 + ["CC"] * 10
        self.sample_meta = pd.DataFrame(
            {"condition": conditions},
            index=self.expression_df.columns,
        )

    def test_baseline_model_initialization(self):
        """Test baseline model initialization."""
        model = BaselineModel(model_type="logistic", random_state=42)

        self.assertEqual(model.model_type, "logistic")
        self.assertFalse(model.is_fitted)

    def test_baseline_model_fit(self):
        """Test baseline model training."""
        model = BaselineModel(model_type="logistic", random_state=42, test_size=0.3)
        model.fit(self.expression_df, self.sample_meta)

        self.assertTrue(model.is_fitted)
        self.assertIsNotNone(model.X_test)
        self.assertIsNotNone(model.y_test)
        self.assertIsNotNone(model.y_pred)

    def test_baseline_model_evaluate(self):
        """Test baseline model evaluation."""
        model = BaselineModel(model_type="logistic", random_state=42)
        model.fit(self.expression_df, self.sample_meta)
        results = model.evaluate()

        self.assertIsInstance(results, BaselineResults)
        self.assertGreaterEqual(results.accuracy, 0.0)
        self.assertLessEqual(results.accuracy, 1.0)
        self.assertIsNotNone(results.confusion_matrix)

    def test_baseline_model_predict(self):
        """Test baseline model prediction."""
        model = BaselineModel(model_type="logistic", random_state=42)
        model.fit(self.expression_df, self.sample_meta)

        predictions = model.predict(self.expression_df)
        self.assertEqual(len(predictions), self.expression_df.shape[1])
        self.assertTrue(all(pred in [0, 1] for pred in predictions))

    def test_train_baseline_convenience(self):
        """Test train_baseline convenience function."""
        model, results = train_baseline(
            self.expression_df,
            self.sample_meta,
            model_type="logistic",
            random_state=42,
        )

        self.assertIsInstance(model, BaselineModel)
        self.assertIsInstance(results, BaselineResults)
        self.assertTrue(model.is_fitted)

    def test_mlp_model(self):
        """Test MLP baseline model."""
        model = BaselineModel(model_type="mlp", random_state=42, test_size=0.3)
        model.fit(self.expression_df, self.sample_meta)

        self.assertTrue(model.is_fitted)
        results = model.evaluate()
        self.assertGreaterEqual(results.accuracy, 0.0)


if __name__ == "__main__":
    unittest.main()
