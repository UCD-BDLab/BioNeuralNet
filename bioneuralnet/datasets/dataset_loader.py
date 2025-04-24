from pathlib import Path
import pandas as pd

class DatasetLoader:
    def __init__(self, dataset_name: str, feature_method: str = "var"):
        """
        Args:
            dataset_name (str): "example1", "monet", or "tcga_brca"
            feature_method (str): for "tcga_brca" only, one of:
                - "var" (variance filter, default)
                - "ae" (autoencoder selection)
                - "anova" (ANOVA F-test selection)
                - "rf" (RandomForest importance selection)
        """
        self.dataset_name = dataset_name.strip().lower()
        self.feature_method = feature_method.strip().lower()
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

        elif self.dataset_name == "tcga_brca":
            valid = {"var", "ae", "anova", "rf"}
            if self.feature_method not in valid:
                raise ValueError(f"For tcga_brca, feature_method must be one of {valid}, but got {self.feature_method}")

            self.data["brca_mirna"]   = pd.read_csv(folder / "brca_mirna.csv",   index_col=0)
            self.data["brca_pam50"]   = pd.read_csv(folder / "brca_pam50.csv",   index_col=0)
            self.data["brca_clinical"] = pd.read_csv(folder / "brca_clinical.csv", index_col=0)

            meth_file = f"brca_meth_{self.feature_method}.csv"
            rna_file  = f"brca_rna_{self.feature_method}.csv"
            self.data["brca_meth"] = pd.read_csv(folder / meth_file, index_col=0)
            self.data["brca_rna"]  = pd.read_csv(folder / rna_file,  index_col=0)
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