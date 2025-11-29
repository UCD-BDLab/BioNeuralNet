import unittest
import pandas as pd
from pathlib import Path
import sys

# Add PD-Notebooks to path
project_root = Path(__file__).resolve().parents[1]
pd_notebooks = project_root / "PD-Notebooks"
if str(pd_notebooks) not in sys.path:
    sys.path.insert(0, str(pd_notebooks))

from bioneuralnet.datasets.parkinsons_loader import (
    ParkinsonsLoader,
    ParkinsonsData,
    load_parkinsons_data,
)


class TestParkinsonsLoader(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.data_dir = project_root / "PD-Notebooks" / "datasets"
        self.counts_file = self.data_dir / "GSE165082_PD-CC.counts.txt"
        self.annotation_file = self.data_dir / "Human.GRCh38.p13.annot.tsv"

    def test_loader_initialization(self):
        """Test loader initialization."""
        loader = ParkinsonsLoader()
        self.assertIsNotNone(loader.counts_path)
        self.assertIsNotNone(loader.annotation_path)

    def test_load_data(self):
        """Test loading PD data."""
        loader = ParkinsonsLoader(use_annotation=True)
        data = loader.load()

        self.assertIsInstance(data, ParkinsonsData)
        self.assertIsInstance(data.expression, pd.DataFrame)
        self.assertIsInstance(data.sample_metadata, pd.DataFrame)
        self.assertIsInstance(data.gene_metadata, pd.DataFrame)

        # Check expression matrix
        self.assertGreater(data.expression.shape[0], 0)  # genes
        self.assertGreater(data.expression.shape[1], 0)  # samples

        # Check sample metadata
        self.assertIn("condition", data.sample_metadata.columns)
        self.assertEqual(len(data.sample_metadata), data.expression.shape[1])

        # Check gene metadata alignment
        self.assertEqual(len(data.gene_metadata), data.expression.shape[0])

    def test_sample_label_parsing(self):
        """Test PD vs Control label parsing."""
        loader = ParkinsonsLoader()
        data = loader.load()

        conditions = data.sample_metadata["condition"].value_counts()
        self.assertIn("PD", conditions.index)
        self.assertIn("CC", conditions.index)

    def test_convenience_function(self):
        """Test load_parkinsons_data convenience function."""
        expr, sample_meta, gene_meta = load_parkinsons_data(use_annotation=True)

        self.assertIsInstance(expr, pd.DataFrame)
        self.assertIsInstance(sample_meta, pd.DataFrame)
        self.assertIsInstance(gene_meta, pd.DataFrame)

        # Check alignment
        self.assertEqual(expr.shape[1], len(sample_meta))
        self.assertEqual(expr.shape[0], len(gene_meta))

    def test_loader_without_annotation(self):
        """Test loader without gene annotations."""
        loader = ParkinsonsLoader(use_annotation=False)
        data = loader.load()

        self.assertIsInstance(data.gene_metadata, pd.DataFrame)
        # Should have empty columns but correct index
        self.assertEqual(len(data.gene_metadata), data.expression.shape[0])


if __name__ == "__main__":
    unittest.main()
