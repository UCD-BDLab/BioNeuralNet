from pathlib import Path
import pandas as pd

class DatasetLoader:
    def __init__(self, dataset_name: str):
        """
        Loads a pre-processed multi-omics dataset from the package.

        Args:

            - dataset_name (str): The name of the dataset to load.
                - Valid options: "example1", "monet", "brca", "gbmlgg", or "kipan".

        Returns:

            - data (dict): Dictionary of DataFrames, where keys are table names and values are DataFrames.

        Example:

            - tcga_kipan = DatasetLoader("kipan")
            - tcga_kipan.shape: {'mirna': (658, 472), 'target': (658, 1), ...}

        """
        self.dataset_name = dataset_name.strip().lower()
        self.base_dir = Path(__file__).parent
        self.data: dict[str, pd.DataFrame] = {}

        self._load_data()

    def _load_data(self):
        """
        Internal loader for the dataset.
        """
        folder = self.base_dir / self.dataset_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset folder '{folder}' not found.")

        if self.dataset_name == "example1":
            self.data = {
                "X1": pd.read_csv(folder / "X1.csv", index_col=0),
                "X2": pd.read_csv(folder / "X2.csv", index_col=0),
                "Y": pd.read_csv(folder / "Y.csv", index_col=0),
                "clinical_data": pd.read_csv(folder / "clinical_data.csv", index_col=0),
            }

        elif self.dataset_name == "monet":
            self.data = {
                "gene_data": pd.read_csv(folder / "gene_data.csv"),
                "mirna_data": pd.read_csv(folder / "mirna_data.csv"),
                "phenotype": pd.read_csv(folder / "phenotype.csv"),
                "rppa_data": pd.read_csv(folder / "rppa_data.csv"),
                "clinical_data": pd.read_csv(folder / "clinical_data.csv"),
            }

        elif self.dataset_name == "brca":
            self.data["mirna"] = pd.read_csv(folder / "mirna.csv", index_col=0)
            self.data["target"] = pd.read_csv(folder / "target.csv", index_col=0)
            self.data["clinical"] = pd.read_csv(folder / "clinical.csv", index_col=0)
            self.data["rna"] = pd.read_csv(folder / "rna.csv", index_col=0)
            self.data["meth"] = pd.read_csv(folder / "meth.csv", index_col=0)

        elif self.dataset_name == "gbmlgg":
            self.data["mirna"] = pd.read_csv(folder / "mirna.csv", index_col=0)
            self.data["target"] = pd.read_csv(folder / "target.csv", index_col=0)
            self.data["clinical"] = pd.read_csv(folder / "clinical.csv", index_col=0)
            self.data["rna"] = pd.read_csv(folder / "rna.csv", index_col=0)
            self.data["meth"] = pd.read_csv(folder / "meth.csv", index_col=0)

        elif self.dataset_name == "kipan":
            self.data["mirna"] = pd.read_csv(folder / "mirna.csv", index_col=0)
            self.data["target"] = pd.read_csv(folder / "target.csv", index_col=0)
            self.data["clinical"] = pd.read_csv(folder / "clinical.csv", index_col=0)
            self.data["rna"] = pd.read_csv(folder / "rna.csv", index_col=0)
            self.data["meth"] = pd.read_csv(folder / "meth.csv", index_col=0)

        elif self.dataset_name == "paad":
            self.data["cnv_amp"] = pd.read_csv(folder / "cnv_amp.csv", index_col=0)
            self.data["cnv_del"] = pd.read_csv(folder / "cnv_del.csv", index_col=0)
            self.data["target"] = pd.read_csv(folder / "target.csv", index_col=0)
            self.data["clinical"] = pd.read_csv(folder / "clinical.csv", index_col=0)
            self.data["rna"] = pd.read_csv(folder / "rna.csv", index_col=0)
            self.data["meth"] = pd.read_csv(folder / "meth.csv", index_col=0)
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' is not recognized.")

    @property
    def shape(self) -> dict[str, tuple[int, int]]:
        """
        dict of table_name to (n_rows, n_cols)
        """
        result = {}
        for name, df in self.data.items():
            result[name] = df.shape
        return result
