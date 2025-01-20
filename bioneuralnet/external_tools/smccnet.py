import os
import subprocess
import pandas as pd
from typing import List, Dict, Any
from ..utils.logger import get_logger
import json


class SmCCNet:
    """
    SmCCNet Class for Graph Generation using Sparse Multiple Canonical Correlation Networks (SmCCNet).

    This class handles the preprocessing of omics data, execution of the SmCCNet R script,
    and retrieval of the resulting adjacency matrix, all using in-memory data structures.
    """

    def __init__(
        self,
        phenotype_df: pd.DataFrame,
        omics_dfs: List[pd.DataFrame],
        data_types: List[str],
        kfold: int = 5,
        summarization: str = "PCA",
        seed: int = 732,
    ):
        """
        Initializes the SmCCNet instance.

        Args:
            phenotype_df (pd.DataFrame): DataFrame containing phenotype data. The first column should be sample IDs.
            omics_dfs (List[pd.DataFrame]): List of DataFrames, each representing an omics dataset. Each DataFrame should have sample IDs as the first column.
            data_types (List[str]): List of omics data types (e.g., ["protein", "metabolite"]).
            kfold (int, optional): Number of folds for cross-validation. Defaults to 5.
            summarization (str, optional): Summarization method. Defaults to "PCA".
            seed (int, optional): Random seed for reproducibility. Defaults to 732.
        """
        self.phenotype_df = phenotype_df
        self.omics_dfs = omics_dfs
        self.data_types = data_types
        self.kfold = kfold
        self.summarization = summarization
        self.seed = seed

        self.logger = get_logger(__name__)
        self.logger.info("Initialized SmCCNet with the following parameters:")
        self.logger.info(f"K-Fold: {self.kfold}")
        self.logger.info(f"Summarization: {self.summarization}")
        self.logger.info(f"Seed: {self.seed}")

        if len(self.omics_dfs) != len(self.data_types):
            self.logger.error(
                "Number of omics dataframes does not match number of data types."
            )
            raise ValueError(
                "Number of omics dataframes does not match number of data types."
            )

    def preprocess_data(self) -> Dict[str, Any]:
        """
        Preprocesses (lightly validates) the phenotype and omics data:
        - Checks that each omics DataFrame has the same number of rows as the phenotype DataFrame.
        - Checks that the first column (sample IDs) match between phenotype and each omics DataFrame.
            If they do not match, logs a warning and replaces them with sequential integer IDs.
        - Serializes the phenotype and each omics DataFrame to CSV.

        Returns:
            Dict[str, Any]: Dictionary containing serialized phenotype and omics data.
                            Keys:
                            "phenotype" : CSV string of the phenotype DataFrame.
                            "omics_1", "omics_2", ...: CSV strings of each omics DataFrame.
        """
        self.logger.info("Validating phenotype and omics data...")

        pheno_df = self.phenotype_df.copy()
        num_samples = len(pheno_df)
        pheno_id_col = pheno_df.columns[0]
        self.logger.info(f"Number of samples in phenotype data: {num_samples}")
        serialized_data = {"phenotype": pheno_df.to_csv(index=False)}

        for i, omics_df in enumerate(self.omics_dfs, start=1):
            current_key = f"omics_{i}"
            df = omics_df.copy()

            if len(df) != num_samples:
                raise ValueError(
                    f"Mismatch in sample count for {current_key}: phenotype has {num_samples} rows, "
                    f"but {current_key} has {len(df)} rows. Please align your data."
                )

            omics_id_col = df.columns[0]

            if not df[omics_id_col].equals(pheno_df[pheno_id_col]):
                self.logger.warning(
                    f"Sample IDs in phenotype '{pheno_id_col}' and {current_key} '{omics_id_col}' do not match. "
                    "Replacing with sequential IDs."
                )

                df[omics_id_col] = range(1, num_samples + 1)
                pheno_df[pheno_id_col] = range(1, num_samples + 1)

            if df.isna().any().any():
                self.logger.warning(f"NaN values found in {current_key}.")
            if ((df == float("inf")) | (df == -float("inf"))).any().any():
                self.logger.warning(f"Inf values found in {current_key}.")

            serialized_data[current_key] = df.to_csv(index=False)

        self.logger.info("Preprocessing checks completed successfully.")
        return serialized_data

    def run_smccnet(self, serialized_data: Dict[str, Any]) -> str:
        """
        Executes the SmCCNet R script by passing serialized data via standard input.

        Args:
            serialized_data (Dict[str, Any]): Dictionary containing serialized phenotype and omics data.

        Returns:
            str: Serialized adjacency matrix JSON string from R script.
        """

        try:
            self.logger.info("Preparing data for SmCCNet R script.")
            json_data = json.dumps(serialized_data) + "\n"

            script_dir = os.path.dirname(os.path.abspath(__file__))
            r_script = os.path.join(script_dir, "SmCCNet.R")

            if not os.path.isfile(r_script):
                self.logger.error(f"R script not found: {r_script}")
                raise FileNotFoundError(f"R script not found: {r_script}")

            command = [
                "Rscript",
                r_script,
                ",".join(self.data_types),
                str(self.kfold),
                self.summarization,
                str(self.seed),
            ]

            self.logger.debug(f"Executing command: {' '.join(command)}")

            result = subprocess.run(
                command, input=json_data, text=True, capture_output=True, check=True
            )

            self.logger.info("SmCCNet R script executed successfully.")
            self.logger.debug(f"Raw R script stdout: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"SmCCNet Warnings/Errors:\n{result.stderr}")

            adjacency_json = result.stdout.strip()
            try:
                json.loads(adjacency_json)
            except json.JSONDecodeError:
                self.logger.error("Invalid JSON output from R script.")
                raise ValueError("R script did not produce valid JSON output.")

            return adjacency_json

        except subprocess.CalledProcessError as e:
            self.logger.error(f"R script execution failed: {e.stderr}")
            raise
        except Exception as e:
            self.logger.error(f"Error during SmCCNet execution: {e}")
            raise

    def run(self) -> pd.DataFrame:
        """
        Executes the entire Sparse Multiple Canonical Correlation Network (SmCCNet) workflow.

        **Steps:**

        1. **Preprocessing Data**:
           - Formats and serializes the input omics and phenotype data for SmCCNet analysis.

        2. **Graph Generation**:
           - Constructs a global network by generating an adjacency matrix through SmCCNet.

        3. **Postprocessing Results**:
           - Deserializes the adjacency matrix (output of SmCCNet) into a Pandas DataFrame.

        **Returns**: pd.DataFrame

            - A DataFrame containing the adjacency matrix, where each entry represents the strength of the correlation between features.

        **Raises**:

            - **ValueError**: If the input data is improperly formatted or missing.
            - **Exception**: For any unforeseen errors encountered during preprocessing, graph generation, or postprocessing.

        **Notes**:

            - SmCCNet is designed for multi-omics data and requires a well-preprocessed and normalized dataset.
            - Ensure that omics and phenotype data are properly aligned to avoid errors in graph construction.

        **Example**:

        .. code-block:: python

            smccnet = SmCCNet(omics_data, phenotype_data)
            adjacency_matrix = smccnet.run()
            print(adjacency_matrix.head())
        """
        try:
            self.logger.info("Starting SmCCNet Graph Generation Workflow.")
            serialized_data = self.preprocess_data()
            adjacency_json = self.run_smccnet(serialized_data)

            self.logger.info("SmCCNet JSON output received.")
            adjacency_dict = json.loads(adjacency_json)
            self.logger.info("type(adjacency_dict): %s", type(adjacency_dict))
            self.logger.info("adjacency_dict.keys(): %s", adjacency_dict)
            adjacency_matrix = pd.DataFrame(
                data=adjacency_dict["data"],
                index=adjacency_dict["index"],
                columns=adjacency_dict["columns"],
            )
            self.logger.info(
                "Adjacency matrix loaded with shape: %s", adjacency_matrix.shape
            )
            self.logger.info("SmCCNet Graph Generation completed successfully.")
            return adjacency_matrix
        except Exception as e:
            self.logger.error(f"Error in SmCCNet Graph Generation: {e}")
            raise
