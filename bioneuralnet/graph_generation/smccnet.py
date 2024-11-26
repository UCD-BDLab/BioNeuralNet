import os
import subprocess
import pandas as pd
from typing import List
from ..utils.logger import get_logger
from datetime import datetime

class SmCCNet:
    """
    SmCCNet Class for Graph Generation using SmCCNet.

    This class handles the execution of SmCCNet R scripts, data preprocessing,
    and loading of the resulting adjacency matrix.

    Attributes:
        phenotype_file (str): Path to phenotype data CSV file.
        omics_list (List[str]): List of paths to omics data CSV files.
        data_types (List[str]): List of omics data types.
        kfold (int): Number of folds for cross-validation.
        summarization (str): Summarization method.
        seed (int): Random seed for reproducibility.
    """

    def __init__(
        self,
        phenotype_file: str,
        omics_list: List[str],
        data_types: List[str],
        kfold: int = 5,
        summarization: str = "PCA",
        seed: int = 732,
    ):
        """
        Initializes the SmCCNet instance with direct parameters.

        Args:
            phenotype_file (str): Path to phenotype data CSV file.
            omics_list (List[str]): List of paths to omics data CSV files.
            data_types (List[str], optional): List of omics data types (e.g., ["protein", "metabolite"]).
                Defaults to ["protein", "metabolite"].
            kfold (int, optional): Number of folds for cross-validation. Defaults to 5.
            summarization (str, optional): Summarization method. Defaults to "PCA".
            seed (int, optional): Random seed for reproducibility. Defaults to 732.
        """
        # Assign parameters
        self.phenotype_file = phenotype_file
        self.omics_list = omics_list
        self.data_types = data_types
        self.kfold = kfold
        self.summarization = summarization
        self.seed = seed

        self.logger = get_logger(__name__)
        self.logger.info("Initialized SmCCNet with the following parameters:")
        self.logger.info(f"Phenotype File: {self.phenotype_file}")
        self.logger.info(f"Omics Files: {self.omics_list}")
        self.logger.info(f"Data Types: {self.data_types}")
        self.logger.info(f"K-Fold: {self.kfold}")
        self.logger.info(f"Summarization: {self.summarization}")
        self.logger.info(f"Seed: {self.seed}")


    def _create_output_dir(self) -> str:
        """
        Creates a unique output directory for the current SmCCNet run.

        The directory is named 'smccnet_output_timestamp' and is created in the current working directory.

        Returns:
            str: Path to the created output directory.
        """
        base_dir = "smccnet_output"
        counter = datetime.now().strftime("%Y%m%d%H%M%S")
        while True:
            output_dir = f"{base_dir}_{counter}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.logger.info(f"Created output directory: {output_dir}")
                return output_dir

    def run(self) -> pd.DataFrame:
        """
        Executes the SmCCNet algorithm and returns the global network adjacency matrix.

        This method orchestrates the preprocessing of data, execution of the SmCCNet R script,
        loading of the resulting adjacency matrix, and cleanup of output files.

        Returns:
            pd.DataFrame: Adjacency matrix representing the global network.

        Raises:
            FileNotFoundError: If essential files are missing.
            subprocess.CalledProcessError: If the R script execution fails.
            Exception: For any other unforeseen errors during execution.
        """
        try:
            self.logger.info("Starting SmCCNet Graph Generation")

            # Create a unique output directory for this run
            output_dir = self._create_output_dir()

            # Preprocess data
            self.preprocess_data()

            # Execute SmCCNet R script
            self.run_smccnet(output_dir)

            # Load the global network adjacency matrix
            adjacency_matrix = self.load_global_network(output_dir)

            self.logger.info("SmCCNet Algorithm executed successfully.")
            return adjacency_matrix
        except Exception as e:
            self.logger.error(f"Error in SmCCNet Graph Generation: {e}")
            raise

    def preprocess_data(self) -> None:
        """
        Preprocesses the combined omics data by checking for NaN or infinite values.

        Steps:
            1. Load all omics data from the specified CSV files.
            2. Ensure that all omics datasets have the same sample IDs and order as the phenotype data.
            3. Remove samples with any NaN or infinite values across all datasets.
            4. Save the cleaned omics data back to their respective CSV files.

        Raises:
            FileNotFoundError: If any omics data file is not found.
            pd.errors.EmptyDataError: If any omics data file is empty.
        """
        self.logger.info("Preprocessing omics data for NaN or infinite values.")

        # Load phenotype data
        try:
            phenotype_data = pd.read_csv(self.phenotype_file, header=True, stringsAsFactors=False)
            self.logger.info(f"Phenotype data loaded with shape {phenotype_data.shape}")
        except FileNotFoundError:
            self.logger.error(f"Phenotype data file not found: {self.phenotype_file}")
            raise
        except pd.errors.EmptyDataError:
            self.logger.error(f"Phenotype data file is empty: {self.phenotype_file}")
            raise
        

        if len(self.omics_list) != len(self.data_types):
            self.logger.error("Number of omics data files does not match number of data types.")
            raise ValueError("Number of omics data files does not match number of data types.")

        # Assuming the first column is sample IDs
        sample_ids_pheno = phenotype_data.iloc[:, 0]
        self.logger.info(f"Number of samples in phenotype data: {len(sample_ids_pheno)}")

        # Initialize a DataFrame to track valid samples
        valid_samples = pd.Series([True] * len(sample_ids_pheno), index=sample_ids_pheno.index)

        # Iterate over each omics dataset
        for omics_file in self.omics_list:
            self.logger.info(f"Processing omics file: {omics_file}")

            # Load omics data
            try:
                omics_data = pd.read_csv(omics_file, header=True, stringsAsFactors=False)
                self.logger.info(f"Omics data loaded with shape {omics_data.shape}")
            except FileNotFoundError:
                self.logger.error(f"Omics data file not found: {omics_file}")
                raise
            except pd.errors.EmptyDataError:
                self.logger.error(f"Omics data file is empty: {omics_file}")
                raise

            # Assuming the first column is sample IDs
            sample_ids_omics = omics_data.iloc[:, 0]
            omics_values = omics_data.iloc[:, 1:]

            # Align the omics data with the phenotype data using sample IDs
            aligned_data = omics_values.set_index(sample_ids_omics).loc[sample_ids_pheno].reset_index(drop=True)

            # Check for any mismatches after alignment
            if aligned_data.isnull().values.any():
                self.logger.warning(f"NaN values detected in omics data after alignment for file: {omics_file}")
                valid_samples &= ~aligned_data.isnull().any(axis=1)

            # Check for infinite values
            if not pd.api.types.is_numeric_dtype(aligned_data.dtypes).all():
                self.logger.warning(f"Non-numeric values detected in omics data for file: {omics_file}. Attempting to convert.")
                aligned_data = aligned_data.apply(pd.to_numeric, errors='coerce')
                if aligned_data.isnull().values.any():
                    self.logger.warning(f"NaN values detected after conversion in omics data for file: {omics_file}")
                    valid_samples &= ~aligned_data.isnull().any(axis=1)

            if (aligned_data == float('inf')).any().any() or (aligned_data == -float('inf')).any().any():
                self.logger.warning(f"Infinite values detected in omics data for file: {omics_file}. Replacing with NaN.")
                aligned_data.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
                valid_samples &= ~aligned_data.isnull().any(axis=1)

            # Save the aligned and cleaned omics data back to the CSV file
            omics_data_clean = pd.concat([sample_ids_pheno, aligned_data], axis=1)
            omics_data_clean.to_csv(omics_file, index=False)
            self.logger.info(f"Cleaned omics data saved to {omics_file}")

        # Determine which samples are valid across all omics datasets
        num_valid_samples = valid_samples.sum()
        self.logger.info(f"Number of valid samples after preprocessing: {num_valid_samples} out of {len(sample_ids_pheno)}")

        if num_valid_samples == 0:
            self.logger.error("No valid samples remaining after preprocessing. Aborting SmCCNet run.")
            raise ValueError("No valid samples remaining after preprocessing.")


    def run_smccnet(self, output_dir: str) -> None:
        """
        Executes the R script for SmCCNet.

        Constructs the command to run the SmCCNet R script with appropriate arguments
        and captures its output.

        Args:
            output_dir (str): Directory where output files will be saved.

        Raises:
            FileNotFoundError: If the SmCCNet R script is not found.
            subprocess.CalledProcessError: If the R script execution fails.
        """
        kfold = self.kfold
        summarization = self.summarization
        seed = self.seed
        data_types_str = ','.join(self.data_types)
        omics_files_str = ','.join(self.omics_list)

        # Determine the path to the R script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        r_script = os.path.join(script_dir, "SmCCNet.R")

        if not os.path.isfile(r_script):
            self.logger.error(f"R script not found: {r_script}")
            raise FileNotFoundError(f"R script not found: {r_script}")

        # Construct the command to execute the R script
        command = [
            "Rscript",
            r_script,
            self.phenotype_file,
            omics_files_str,
            data_types_str,
            str(kfold),
            summarization,
            str(seed),
            output_dir,
        ]

        self.logger.debug(f"Executing command: {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.logger.info("SmCCNet R script executed successfully.")
            self.logger.debug(f"SmCCNet Output:\n{result.stdout}")

            if result.stderr:
                self.logger.warning(f"SmCCNet Warnings/Errors:\n{result.stderr}")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"R script execution failed: {e.stderr}")
            raise

    def load_global_network(self, output_dir: str) -> pd.DataFrame:
        """
        Loads the global network adjacency matrix generated by SmCCNet.

        Args:
            output_dir (str): Directory where output files are saved.

        Returns:
            pd.DataFrame: Adjacency matrix of the global network.

        Raises:
            FileNotFoundError: If the global network CSV file is not found.
            pd.errors.EmptyDataError: If the CSV file is empty.
        """
        global_network_csv = os.path.join(output_dir, "global_network.csv")

        if not os.path.isfile(global_network_csv):
            self.logger.error(f"Global network file not found: {global_network_csv}")
            raise FileNotFoundError(f"Global network file not found: {global_network_csv}")

        # Load the adjacency matrix
        try:
            adjacency_matrix = pd.read_csv(global_network_csv, index_col=0)
            self.logger.info("Global network adjacency matrix loaded successfully.")
            return adjacency_matrix
        except pd.errors.EmptyDataError:
            self.logger.error(f"Global network CSV file is empty: {global_network_csv}")
            raise
