import os
import subprocess
import pandas as pd
from typing import List, Dict, Any
from ..utils.logger import get_logger
import json
from io import StringIO


class WGCNA:
    """
    WGCNA Class for Graph Construction using Weighted Gene Co-expression Network Analysis (WGCNA).

    This class handles the preprocessing of omics data, execution of the WGCNA R script,
    and retrieval of the resulting adjacency matrix, all using in-memory data structures.
    """

    def __init__(
        self,
        phenotype_df: pd.DataFrame,
        omics_dfs: List[pd.DataFrame],
        data_types: List[str],
        soft_power: int = 6,
        min_module_size: int = 30,
        merge_cut_height: float = 0.25,
    ):
        """
        Initializes the WGCNA instance.

        Args:
            phenotype_df (pd.DataFrame): DataFrame containing phenotype data. The first column should be sample IDs.
            omics_dfs (List[pd.DataFrame]): List of DataFrames, each representing an omics dataset. Each DataFrame should have sample IDs as the first column.
            data_types (List[str]): List of data types corresponding to each omics dataset.
            soft_power (int, optional): Soft-thresholding power. Defaults to 6.
            min_module_size (int, optional): Minimum module size. Defaults to 30.
            merge_cut_height (float, optional): Merge cut height. Defaults to 0.25.
        """
        self.phenotype_df = phenotype_df
        self.omics_dfs = omics_dfs
        self.data_types = data_types
        self.soft_power = soft_power
        self.min_module_size = min_module_size
        self.merge_cut_height = merge_cut_height

        self.logger = get_logger(__name__)
        self.logger.info("Initialized WGCNA with the following parameters:")
        self.logger.info(f"Soft Power: {self.soft_power}")
        self.logger.info(f"Minimum Module Size: {self.min_module_size}")
        self.logger.info(f"Merge Cut Height: {self.merge_cut_height}")

        if len(self.omics_dfs) != len(self.data_types):
            self.logger.error(
                "Number of omics dataframes does not match number of data types."
            )
            raise ValueError(
                "Number of omics dataframes does not match number of data types."
            )

    def preprocess_data(self) -> Dict[str, Any]:
        """
        Preprocesses the omics data to ensure alignment and handle missing values.

        Returns:
            Dict[str, Any]: Dictionary containing serialized phenotype and omics data.
        """
        self.logger.info("Preprocessing omics data for NaN or infinite values.")
        phenotype_ids = self.phenotype_df.iloc[:, 0]
        self.logger.info(f"Number of samples in phenotype data: {len(phenotype_ids)}")
        valid_samples = pd.Series(
            [True] * len(phenotype_ids), index=self.phenotype_df.index
        )

        serialized_data = {"phenotype": self.phenotype_df.to_csv(index=False)}

        for idx, omics_df in enumerate(self.omics_dfs):
            data_type = self.data_types[idx]
            self.logger.info(
                f"Processing omics DataFrame {idx+1}/{len(self.omics_dfs)}: Data Type = {data_type}"
            )

            omics_ids = omics_df.iloc[:, 0]
            if not omics_ids.equals(phenotype_ids):
                self.logger.warning(
                    f"Sample IDs in omics dataframe {idx+1} do not match phenotype data. Aligning data."
                )
                omics_df = (
                    omics_df.set_index(omics_ids).loc[phenotype_ids].reset_index()
                )

            if omics_df.isnull().values.any():
                self.logger.warning(
                    f"NaN values detected in omics dataframe {idx+1}. Marking samples with NaNs as invalid."
                )
                valid_samples &= ~omics_df.isnull().any(axis=1)

            if (omics_df == float("inf")).any().any() or (
                omics_df == -float("inf")
            ).any().any():
                self.logger.warning(
                    f"Infinite values detected in omics dataframe {idx+1}. Replacing with NaN and marking samples as invalid."
                )
                omics_df.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
                valid_samples &= ~omics_df.isnull().any(axis=1)

            num_valid_before = valid_samples.sum()
            self.logger.info(
                f"Number of valid samples before filtering: {num_valid_before}"
            )
            omics_df_clean = omics_df[valid_samples].reset_index(drop=True)
            num_valid_after = omics_df_clean.shape[0]
            self.logger.info(
                f"Number of valid samples after filtering: {num_valid_after}"
            )

            if num_valid_after == 0:
                self.logger.error(
                    "No valid samples remaining after preprocessing. Aborting WGCNA run."
                )
                raise ValueError("No valid samples remaining after preprocessing.")

            serialized_data[f"omics_{idx+1}"] = omics_df_clean.to_csv(index=False)

        self.logger.info("Preprocessing completed successfully.")
        return serialized_data

    def run_wgcna(self, serialized_data: Dict[str, Any]) -> str:
        """
        Executes the WGCNA R script by passing serialized data via standard input.

        Args:
            serialized_data (Dict[str, Any]): Dictionary containing serialized phenotype and omics data.

        Returns:
            str: Serialized adjacency matrix CSV string from R script.
        """
        try:
            self.logger.info("Preparing data for WGCNA R script.")
            json_data = json.dumps(serialized_data)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            r_script = os.path.join(script_dir, "WGCNA.R")

            if not os.path.isfile(r_script):
                self.logger.error(f"R script not found: {r_script}")
                raise FileNotFoundError(f"R script not found: {r_script}")

            command = [
                "Rscript",
                r_script,
                str(self.soft_power),
                str(self.min_module_size),
                str(self.merge_cut_height),
            ]

            self.logger.debug(f"Executing command: {' '.join(command)}")
            result = subprocess.run(
                command, input=json_data, text=True, capture_output=True, check=True
            )

            self.logger.info("WGCNA R script executed successfully.")
            self.logger.debug(f"WGCNA Output:\n{result.stdout}")

            if result.stderr:
                self.logger.warning(f"WGCNA Warnings/Errors:\n{result.stderr}")

            adjacency_json = result.stdout.strip()

            return adjacency_json

        except subprocess.CalledProcessError as e:
            self.logger.error(f"R script execution failed: {e.stderr}")
            raise
        except Exception as e:
            self.logger.error(f"Error during WGCNA execution: {e}")
            raise

    def run(self) -> pd.DataFrame:
        """
        Executes the entire Weighted Gene Co-expression Network Analysis (WGCNA) workflow.

        **Steps:**

        1. **Preprocessing Data:**
            - Prepares and formats the input omics data for WGCNA analysis.
            - Serializes the data into a format suitable for the WGCNA pipeline.

        2. **Running WGCNA:**
            - Constructs a weighted correlation network based on the serialized omics data.
            - Identifies co-expression modules among genes or features in the dataset.

        3. **Postprocessing Results:**
            - Deserializes the adjacency matrix (output of WGCNA) into a Pandas DataFrame.
            - Logs successful completion and prepares the matrix for downstream tasks.

        **Returns**: pd.DataFrame

            - A DataFrame containing the adjacency matrix, where each entry represents the weighted correlation between features.

        **Raises**:

            - ValueError If the input data is improperly formatted or missing.
            - Exception For any unforeseen errors encountered during preprocessing, network construction, or postprocessing.

        **Notes**:

            - The WGCNA workflow is sensitive to input data quality and formatting.
            - Ensure that the input omics data is preprocessed, normalized, and properly indexed to align with expected formats.
            - This method is designed for large-scale multi-omics data and may require significant computational resources depending on the dataset size.

        Example:

        .. code-block:: python

            wgcna = WGCNA(omics_data)
            adjacency_matrix = wgcna.run()
            print(adjacency_matrix.head())
        """
        try:
            self.logger.info("Starting WGCNA Network Construction Workflow.")

            serialized_data = self.preprocess_data()
            adjacency_json = self.run_wgcna(serialized_data)
            adjacency_matrix = pd.read_json(StringIO(adjacency_json), orient="split")

            self.logger.info("Adjacency matrix deserialized successfully.")
            self.logger.info("WGCNA Network Construction completed successfully.")
            return adjacency_matrix

        except Exception as e:
            self.logger.error(f"Error in WGCNA Network Construction: {e}")
            raise
