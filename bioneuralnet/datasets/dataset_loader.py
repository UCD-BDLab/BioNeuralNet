from pathlib import Path
import pandas as pd

class DatasetLoader:
    def __init__(self, dataset_name: str):
        """
        Args:
            dataset_name (str): "example1", "monet", or "tcga_brca".

        returns:

            - data (dict): Dictionary of DataFrames, where keys are table names and values are DataFrames.
            - shape (dict): Dictionary of table names to their shapes (n_rows, n_cols).

        Example:

            tcga_brca = DatasetLoader("tcga_brca")
            tcga_brca.shape
            # {'brca_mirna': (108, 1000), 'brca_pam50': (108, 50), ...}
            mirna = tcga_brca.data["brca_mirna"]
            rna = tcga_brca.data["brca_rna"]

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
            self.data["pam50"] = pd.read_csv(folder / "pam50.csv", index_col=0)
            self.data["clinical"] = pd.read_csv(folder / "clinical.csv", index_col=0)
            self.data["rna"] = pd.read_csv(folder / "rna.csv", index_col=0)
            self.data["meth"] = pd.read_csv(folder / "meth.csv", index_col=0)
            #meth_part1 = pd.read_csv(folder / "meth_1.csv", index_col=0)
            #meth_part2= pd.read_csv(folder / "meth_2.csv", index_col=0)

            #rna_part1 = pd.read_csv(folder / "rna_1.csv", index_col=0)
            #rna_part2 = pd.read_csv(folder / "rna_2.csv", index_col=0)

            #self.data["meth"] = pd.concat([meth_part1, meth_part2], axis=0)
            #self.data["rna"] = pd.concat([rna_part1, rna_part2], axis=0)

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
