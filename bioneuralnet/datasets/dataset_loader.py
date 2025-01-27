import os
import pandas as pd
from typing import Tuple, Optional


class DatasetLoader:
    """
    This class allows users to load predefined sample datasets by specifying
    the dataset name. Each dataset should contain three CSV files:
    - omics1.csv: Genomic features
    - omics2.csv: miRNA features
    - pheno.csv: Phenotype data with a single 'Pheno' column representing the target outcome
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initializes the DatasetLoader with the base data directory.
        """
        if base_dir is None:
            self.base_dir = os.path.join(os.path.dirname(__file__), "example1")
        else:
            self.base_dir = base_dir

    def __call__(self, name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads the specified dataset.
        Args: name (str): Name of the dataset to load (corresponds to a subdirectory in the datasets folder).
        Returns: tuple: A tuple containing the omics1, omics2, and pheno DataFrames.
        Raises: FileNotFoundError if the required files are not found.
        """
        dataset_path = os.path.join(os.path.dirname(__file__), name)
        if not os.path.isdir(dataset_path):
            raise ValueError(
                f"Dataset '{name}' not found in '{os.path.dirname(__file__)}'."
            )

        omics1_path = os.path.join(dataset_path, "X1.csv")
        omics2_path = os.path.join(dataset_path, "X2.csv")
        pheno_path = os.path.join(dataset_path, "Y.csv")

        for path in [omics1_path, omics2_path, pheno_path]:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Required file '{os.path.basename(path)}' not found in '{dataset_path}'."
                )

        omics1 = pd.read_csv(omics1_path, index_col=0)
        omics2 = pd.read_csv(omics2_path, index_col=0)
        pheno = pd.read_csv(pheno_path, index_col=0)

        return omics1, omics2, pheno
