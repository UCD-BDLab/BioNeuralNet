import os
import subprocess
import pandas as pd
from pathlib import Path
import json
import tempfile
from typing import List, Dict, Any
from ..utils.logger import get_logger
import shutil


class SmCCNet:
    """
    SmCCNet Class for Graph Generation using Sparse Multiple Canonical Correlation Networks (SmCCNet).

    This class handles the preprocessing of omics data, execution of the SmCCNet R script,
    and retrieval of the resulting adjacency matrix from a designated output directory.
    """

    def __init__(
        self,
        phenotype_df: pd.DataFrame,
        omics_dfs: List[pd.DataFrame],
        data_types: List[str],
        kfold: int = 5,
        summarization: str = "NetSHy",
        seed: int = 732,
        output_dir: str = None,
    ):
        """
        Initializes the SmCCNet instance.

        Args:
            phenotype_df (pd.DataFrame): DataFrame containing phenotype data.
            omics_dfs (List[pd.DataFrame]): List of omics DataFrames.
            data_types (List[str]): List of omics data types (e.g., ["protein", "metabolite"]).
            kfold (int, optional): Number of folds for cross-validation. Defaults to 5.
            summarization (str, optional): Summarization method. Defaults to "NetSHy".
            seed (int, optional): Random seed. Defaults to 732.
            output_dir (str, optional): Folder to write temporary files. If None, a temporary directory is created.
        """
        self.phenotype_df = phenotype_df
        self.omics_dfs = omics_dfs
        self.data_types = data_types
        self.kfold = kfold
        self.summarization = summarization
        self.seed = seed

        self.logger = get_logger(__name__)
        self.logger.info("Initialized SmCCNet with parameters:")
        self.logger.info(f"K-Fold: {self.kfold}")
        self.logger.info(f"Summarization: {self.summarization}")
        self.logger.info(f"Seed: {self.seed}")

        if len(self.omics_dfs) != len(self.data_types):
            self.logger.error(
                "Number of omics DataFrames does not match number of data types."
            )
            raise ValueError("Mismatch between omics dataframes and data types.")

        if output_dir is None:
            # Use TemporaryDirectory context manager for automatic cleanup
            self.temp_dir_obj = tempfile.TemporaryDirectory()
            self.output_dir = self.temp_dir_obj.name
            self.logger.info(
                f"No output_dir provided; using temporary directory: {self.output_dir}"
            )
        else:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)

    def preprocess_data(self) -> Dict[str, Any]:
        """
        Preprocess the phenotype and omics data:
         - Reset indexes, standardize sample IDs, and serialize to CSV.

        Returns:
            Dict[str, Any]: A dictionary with keys 'phenotype', 'omics_1', etc.
        """
        self.logger.info("Validating and serializing input data for SmCCNet...")
        pheno_df = (
            self.phenotype_df.copy().reset_index().rename(columns={"index": "SampleID"})
        )
        pheno_df["SampleID"] = pheno_df["SampleID"].astype(str).str.strip().str.upper()
        serialized_data = {"phenotype": pheno_df.to_csv(index=False)}

        for i, omics_df in enumerate(self.omics_dfs, start=1):
            key = f"omics_{i}"
            df = omics_df.copy().reset_index().rename(columns={"index": "SampleID"})
            df["SampleID"] = df["SampleID"].astype(str).str.strip().str.upper()
            # Intersect sample IDs between phenotype and omics data
            common_ids = set(pheno_df["SampleID"]).intersection(set(df["SampleID"]))
            if not common_ids:
                raise ValueError(
                    f"No overlapping sample IDs between phenotype and {key}."
                )
            df = df[df["SampleID"].isin(common_ids)]
            # Ensure ordering follows phenotype data
            df = df.set_index("SampleID").loc[pheno_df["SampleID"]].reset_index()
            serialized_data[key] = df.to_csv(index=False)
            self.logger.info(f"Serialized {key} with {len(df)} samples.")
        return serialized_data

    def run_smccnet(self, serialized_data: Dict[str, Any]) -> None:
        """
        Executes the SmCCNet R script in the specified output directory.

        Args:
            serialized_data (Dict[str, Any]): Serialized CSV strings for phenotype and omics data.
        """
        try:
            self.logger.info("Executing SmCCNet R script...")
            json_data = json.dumps(serialized_data) + "\n"
            script_dir = os.path.dirname(os.path.abspath(__file__))

            r_script = os.path.join(script_dir, "SmCCNet.R")
            if not os.path.isfile(r_script):
                self.logger.error(f"R script not found: {r_script}")
                raise FileNotFoundError(f"R script not found: {r_script}")

            # Dynamically locate Rscript in the system path
            rscript_path = shutil.which("Rscript")
            if rscript_path is None:
                raise EnvironmentError("Rscript not found in system PATH.")

            command = [
                rscript_path,
                r_script,
                ",".join(self.data_types),
                str(self.kfold),
                self.summarization,
                str(self.seed),
            ]
            self.logger.debug(
                f"Running command: {' '.join(command)} in cwd={self.output_dir}"
            )
            result = subprocess.run(
                command,
                input=json_data,
                text=True,
                capture_output=True,
                check=True,
                cwd=self.output_dir,
            )
            self.logger.info(f"SMCCNET R script output:\n{result.stdout}")
            if result.stderr:
                self.logger.warning(
                    f"SMCCNET R script warnings/errors:\n{result.stderr}"
                )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"R script execution failed: {e.stderr}")
            raise
        except Exception as e:
            self.logger.error(f"Error during SmCCNet execution: {e}")
            raise

    def run(self) -> pd.DataFrame:
        """
        Runs the full SmCCNet workflow and returns the generated adjacency matrix.

        Returns:
            pd.DataFrame: The adjacency matrix.
        """
        try:
            self.logger.info("Starting SmCCNet workflow.")
            serialized_data = self.preprocess_data()
            self.run_smccnet(serialized_data)
            adjacency_path = Path(self.output_dir) / "AdjacencyMatrix.csv"
            self.logger.info(f"Reading adjacency matrix from: {adjacency_path}")
            adjacency_df = pd.read_csv(adjacency_path, index_col=0)
            self.logger.info(f"Adjacency matrix shape: {adjacency_df.shape}")
            return adjacency_df
        except Exception as e:
            self.logger.error(f"Error in SmCCNet workflow: {e}")
            raise
