from pathlib import Path
import pandas as pd

class DatasetLoader:
    """Load a pre-packaged multi-omics dataset from the package.

    Options for 'dataset_name':

        "example": Synthetic example.
        "monet": Synthetic example.
        "brca": Breast invasive carcinoma.
        "lgg": Brain Lower Grade Glioma.
        "kipan": Pan-kidney carcinoma.

    Args:

        dataset_name (str): Normalized dataset name.
        base_dir (Path): Directory where the dataset folders live.
        data (dict[str, pd.DataFrame]): Mapping from table name to loaded DataFrame.


    """
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name.strip().lower()
        self.base_dir = Path(__file__).parent
        self.data: dict[str, pd.DataFrame] = {}

        self._load_data()

    def __getitem__(self, key):
        return self.data[key]

    def _load_data(self):
        """Internal helper to populate ``self.data`` from CSV files for the given dataset."""
        folder = self.base_dir / self.dataset_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset folder '{folder}' not found.")

        if self.dataset_name == "example":
            self.data = {
                "X1": pd.read_csv(folder / "X1.csv", index_col=0),
                "X2": pd.read_csv(folder / "X2.csv", index_col=0),
                "Y": pd.read_csv(folder / "Y.csv", index_col=0),
                "clinical": pd.read_csv(folder / "clinical.csv", index_col=0),
            }

        elif self.dataset_name == "monet":
            self.data = {
                "gene": pd.read_csv(folder / "gene.csv"),
                "mirna": pd.read_csv(folder / "mirna.csv"),
                "phenotype": pd.read_csv(folder / "phenotype.csv"),
                "rppa": pd.read_csv(folder / "rppa.csv"),
                "clinical": pd.read_csv(folder / "clinical.csv"),
            }

        elif self.dataset_name == "brca":
            self.data["mirna"] = pd.read_csv(folder / "mirna.csv", index_col=0)
            self.data["target"] = pd.read_csv(folder / "target.csv", index_col=0)
            self.data["clinical"] = pd.read_csv(folder / "clinical.csv", index_col=0)
            self.data["rna"] = pd.read_csv(folder / "rna.csv", index_col=0)
            self.data["methylation"] = pd.read_csv(folder / "methylation.csv", index_col=0)

        elif self.dataset_name == "lgg":
            self.data["mirna"] = pd.read_csv(folder / "mirna.csv", index_col=0)
            self.data["target"] = pd.read_csv(folder / "target.csv", index_col=0)
            self.data["clinical"] = pd.read_csv(folder / "clinical.csv", index_col=0)
            self.data["rna"] = pd.read_csv(folder / "rna.csv", index_col=0)
            self.data["methylation"] = pd.read_csv(folder / "methylation.csv", index_col=0)

        elif self.dataset_name == "kipan":
            self.data["mirna"] = pd.read_csv(folder / "mirna.csv", index_col=0)
            self.data["target"] = pd.read_csv(folder / "target.csv", index_col=0)
            self.data["clinical"] = pd.read_csv(folder / "clinical.csv", index_col=0)
            self.data["rna"] = pd.read_csv(folder / "rna.csv", index_col=0)
            self.data["methylation"] = pd.read_csv(folder / "methylation.csv", index_col=0)

        else:
            raise ValueError(f"Dataset '{self.dataset_name}' is not recognized.")

    @property
    def shape(self) -> dict[str, tuple[int, int]]:
        """Dictionary mapping each table name to its (n_rows, n_cols) shape."""
        result: dict[str, tuple[int, int]] = {}
        for name, df in self.data.items():
            result[name] = df.shape
        return result
