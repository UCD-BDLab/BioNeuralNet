import unittest
import pandas as pd
from pathlib import Path
from bioneuralnet.datasets.dataset_loader import DatasetLoader

class TestDatasetLoader(unittest.TestCase):
    def test_example1_loads(self):
        loader = DatasetLoader("example1")
        keys = set(loader.data.keys())
        self.assertEqual(keys, {"X1", "X2", "Y", "clinical_data"})

        for df in loader.data.values():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(df.shape[0], 0)
            self.assertGreater(df.shape[1], 0)

        for name, shape in loader.shape.items():
            self.assertIsInstance(shape, tuple)
            self.assertEqual(len(shape), 2)

    def test_monet_loads(self):
        loader = DatasetLoader("monet")
        keys = set(loader.data.keys())

        self.assertEqual(keys, {"gene_data", "mirna_data", "phenotype", "rppa_data", "clinical_data"})

        for df in loader.data.values():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(df.shape[0], 0)
            self.assertGreater(df.shape[1], 0)

        for name, shape in loader.shape.items():
            self.assertIsInstance(shape, tuple)
            self.assertEqual(len(shape), 2)

    def test_brca_loads(self):
        loader = DatasetLoader("brca")
        keys = set(loader.data.keys())
        self.assertEqual(keys, {"mirna", "pam50", "clinical", "rna", "meth"})
        for df in loader.data.values():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(df.shape[0], 0)
            self.assertGreater(df.shape[1], 0)
        for name, shape in loader.shape.items():
            self.assertIsInstance(shape, tuple)
            self.assertEqual(len(shape), 2)

    def test_invalid_folder_raises(self):
        with self.assertRaises(FileNotFoundError):
            DatasetLoader("nonexistent_folder")

    def test_unrecognized_name_raises(self):
        base = Path(__file__).parent.parent / "bioneuralnet" / "datasets"
        dummy = base / "dummy"
        dummy.mkdir(exist_ok=True)

        (dummy / "placeholder.csv").write_text("a,b\n1,2")

        with self.assertRaises(ValueError):
            DatasetLoader("dummy")

        for child in dummy.iterdir():
            child.unlink()
        dummy.rmdir()

if __name__ == "__main__":
    unittest.main()
