from pathlib import Path
import pandas as pd
from typing import List

class NetworkLoader:
    """
    Class to load bundled networks from the networks folder.
    Current options are:
        - brca_smccnet_ae
        - brca_smccnet_rf
        - brca_smccnet_var

    Networks must live in subfolders each containing:
        - GlobalNetwork.csv
        - size_<n>_net_<i>.csv
    """
    def __init__(self):
        self.base_dir = Path(__file__).parent / "networks"
        if not self.base_dir.is_dir():
            raise FileNotFoundError(f"Bundled networks folder not found at: {self.base_dir}")

        methods: List[str] = []
        for p in self.base_dir.iterdir():
            if p.is_dir():
                methods.append(p.name)

        self.methods = methods

    def available_methods(self) -> List[str]:
        """Return list of bundled network-method names"""
        return self.methods

    def load_global_network(self, method: str) -> pd.DataFrame:
        """
        Load the GlobalNetwork.csv for the given method.
        """

        folder = self.base_dir / method
        path = folder / "GlobalNetwork.csv"
        
        if not path.is_file():
            raise FileNotFoundError(f"GlobalNetwork.csv not found for method {method}")
        
        return pd.read_csv(path, index_col=0)

    def load_clusters(self, method: str) -> List[pd.DataFrame]:
        """
        Load all size_*_net_*.csv cluster files for the given method,
        sorted by (size, index), and return them as DataFrames
        """
        folder = self.base_dir / method
        if not folder.is_dir():
            raise FileNotFoundError(f"Method folder '{method}' not found under {self.base_dir}")

        raw = list(folder.glob("size_*_net_*.csv"))
        sorted_paths: List[Path] = []

        for p in raw:
            parts = p.stem.split("_")
            size = int(parts[1])
            idx  = int(parts[-1])
            inserted = False
            for i, ex in enumerate(sorted_paths):
                ex_parts = ex.stem.split("_")
                ex_size = int(ex_parts[1])
                ex_idx  = int(ex_parts[-1])
                if (size, idx) < (ex_size, ex_idx):
                    sorted_paths.insert(i, p)
                    inserted = True
                    break

            if not inserted:
                sorted_paths.append(p)

        clusters: List[pd.DataFrame] = []
        for path in sorted_paths:
            clusters.append(pd.read_csv(path, index_col=0))

        return clusters
