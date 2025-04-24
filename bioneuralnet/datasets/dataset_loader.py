from pathlib import Path
import pandas as pd

class DatasetLoader:
    def __init__(self, dataset_name: str):
        """
        Args:
            dataset_name (str): "example1", "monet", or "tcga_brca"
        """
        self.dataset_name = dataset_name.strip().lower()
        self.base_dir = Path(__file__).parent
        self.data: dict[str, pd.DataFrame] = {}
        
        self._load_data()

    def _load_and_concat(self, folder: Path, stem: str) -> pd.DataFrame:
        p1 = folder / f"{stem}_part1.csv"
        p2 = folder / f"{stem}_part2.csv"
        if p1.exists() and p2.exists():
            df1 = pd.read_csv(p1, index_col=0)
            df2 = pd.read_csv(p2, index_col=0)
            return pd.concat([df1, df2], axis=0)

        single = folder / f"{stem}.csv"
        if not single.exists():
            raise FileNotFoundError(f"File '{single.name}' not found in '{folder}'.")
        
        return pd.read_csv(single, index_col=0)

    def _load_data(self):
        """
        Internal loader that fills self.data immediately.
        """
        folder = self.base_dir / self.dataset_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset folder '{folder}' not found.")

        if self.dataset_name == "example1":
            self.data = {
                "X1": pd.read_csv(folder / "X1.csv",            index_col=0),
                "X2": pd.read_csv(folder / "X2.csv",            index_col=0),
                "Y": pd.read_csv(folder / "Y.csv",             index_col=0),
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
            self.data = {
                "BRCA_miRNA": pd.read_csv(folder / "BRCA_miRNA.csv",    index_col=0),
                "BRCA_Meth": self._load_and_concat(folder, "BRCA_Meth"),
                "BRCA_RNA": self._load_and_concat(folder, "BRCA_RNA"),
                "BRCA_PAM50": pd.read_csv(folder / "BRCA_PAM50.csv",    index_col=0),
                "BRCA_Clinical": pd.read_csv(folder / "BRCA_Clinical.csv", index_col=0),
            }

        else:
            raise ValueError(f"Dataset '{self.dataset_name}' is not recognized.")

    @property
    def shape(self) -> dict[str, tuple[int,int]]:
        """
        dict of table_name to (n_rows, n_cols), already loaded in __init__.
        """
        result = {}
        for name, df in self.data.items():
            result[name] = df.shape

        return result

