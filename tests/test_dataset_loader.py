import unittest
import pandas as pd
from pathlib import Path
from bioneuralnet.datasets import (
    DatasetLoader,
    load_example,
    load_monet,
    load_brca,
    load_lgg,
    load_kipan,
)

class TestDatasetLoader(unittest.TestCase):
    def test_example_loads(self):
        loader = DatasetLoader("example")
        keys = set(loader.data.keys())
        self.assertEqual(keys, {"X1", "X2", "Y", "clinical"})

        for df in loader.data.values():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(df.shape[0], 0)
            self.assertGreater(df.shape[1], 0)

    def test_monet_loads(self):
        loader = DatasetLoader("monet")
        keys = set(loader.data.keys())
        self.assertEqual(keys, {"gene", "mirna", "phenotype", "rppa", "clinical"})

        for df in loader.data.values():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(df.shape[0], 0)
            self.assertGreater(df.shape[1], 0)

    def test_brca_loads(self):
        loader = DatasetLoader("brca")
        keys = set(loader.data.keys())
        self.assertEqual(keys, {"mirna", "target", "clinical", "rna", "methylation"})

        for df in loader.data.values():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(df.shape[0], 0)
            self.assertGreater(df.shape[1], 0)

    def test_lgg_loads(self):
        loader = DatasetLoader("lgg")
        keys = set(loader.data.keys())
        self.assertEqual(keys, {"mirna", "target", "clinical", "rna", "methylation"})

        for df in loader.data.values():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(df.shape[0], 0)
            self.assertGreater(df.shape[1], 0)

    def test_kipan_loads(self):
        loader = DatasetLoader("kipan")
        keys = set(loader.data.keys())
        self.assertEqual(keys, {"mirna", "target", "clinical", "rna", "methylation"})

        for df in loader.data.values():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(df.shape[0], 0)
            self.assertGreater(df.shape[1], 0)

    def test_getitem_access(self):
        loader = DatasetLoader("example")
        df = loader["X1"]
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (358, 500))

    def test_functional_loaders(self):
        self.assertIsInstance(load_example(), dict)
        self.assertIsInstance(load_monet(), dict)
        self.assertIsInstance(load_brca(), dict)
        self.assertIsInstance(load_lgg(), dict)
        self.assertIsInstance(load_kipan(), dict)
        self.assertEqual(load_brca().keys(), DatasetLoader("brca").data.keys())

    def test_invalid_folder_raises(self):
        with self.assertRaises(FileNotFoundError):
            DatasetLoader("nonexistent_folder")

if __name__ == "__main__":
    unittest.main()
