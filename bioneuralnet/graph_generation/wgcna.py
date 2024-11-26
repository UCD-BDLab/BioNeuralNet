import os
import subprocess
import pandas as pd
from typing import List, Optional
from ..utils.logger import get_logger
from datetime import datetime

class WGCNA:
    """
    WGCNA Class for Graph Construction using Weighted Gene Co-expression Network Analysis (WGCNA).

    This class handles the execution of WGCNA R scripts, data preprocessing,
    and loading of the resulting adjacency matrix.
    
    Attributes:
        phenotype_file (str): Path to phenotype data CSV file.
        omics_list (List[str]): List of paths to omics data CSV files.
        data_types (List[str]): List of omics data types.
        soft_power (int): Soft-thresholding power.
        min_module_size (int): Minimum module size.
        merge_cut_height (float): Module merging threshold.
        output_dir (str): Directory to save outputs.
    """

    def __init__(
        self,
        phenotype_file: str,
        omics_list: List[str],
        data_types: List[str],
        soft_power: int = 6,
        min_module_size: int = 30,
        merge_cut_height: float = 0.25,
        output_dir: Optional[str] = None,
    ):
        """
        Initializes the WGCNA instance with direct parameters.

        Args:
            phenotype_file (str): Path to phenotype data CSV file.
            omics_list (List[str]): List of paths to omics data CSV files.
            data_types (List[str]): List of omics data types (e.g., ["gene", "miRNA"]).
            soft_power (int, optional): Soft-thresholding power. Defaults to 6.
            min_module_size (int, optional): Minimum module size. Defaults to 30.
            merge_cut_height (float, optional): Module merging threshold. Defaults to 0.25.
            output_dir (str, optional): Directory to save outputs. If None, creates a unique directory.
        """
        # Assign parameters
        self.phenotype_file = phenotype_file
        self.omics_list = omics_list
        self.data_types = data_types
        self.soft_power = soft_power
        self.min_module_size = min_module_size
        self.merge_cut_height = merge_cut_height
        self.output_dir = output_dir if output_dir else self._create_output_dir()

        # Initialize logger (global logger)
        self.logger = get_logger(__name__)
        self.logger.info("Initialized WGCNA with the following parameters:")
        self.logger.info(f"Phenotype File: {self.phenotype_file}")
        self.logger.info(f"Omics Files: {self.omics_list}")
        self.logger.info(f"Data Types: {self.data_types}")
        self.logger.info(f"Soft Power: {self.soft_power}")
        self.logger.info(f"Minimum Module Size: {self.min_module_size}")
        self.logger.info(f"Merge Cut Height: {self.merge_cut_height}")
        self.logger.info(f"Output Directory: {self.output_dir}")

    def _create_output_dir(self) -> str:
        """
        Creates a unique output directory for the current WGCNA run.

        The directory is named 'wgcna_output_timestamp' and is created in the current working directory.

        Returns:
            str: Path to the created output directory.
        """
        base_dir = "wgcna_output"
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_dir = f"{base_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Created output directory: {output_dir}")
        return output_dir

    def run(self) -> pd.DataFrame:
        """
        Executes the WGCNA pipeline and returns the global network adjacency matrix.

        This method orchestrates the preprocessing of data, execution of the WGCNA R script,
        loading of the resulting adjacency matrix, and cleanup of output files.

        Returns:
            pd.DataFrame: Adjacency matrix representing the global network.

        Raises:
            FileNotFoundError: If essential files are missing.
            subprocess.CalledProcessError: If the R script execution fails.
            Exception: For any other unforeseen errors during execution.
        """
        try:
            self.logger.info("Starting WGCNA Network Construction")
            self.preprocess_data()
            self.run_wgcna()
            adjacency_matrix = self.load_global_network()
            self.cleanup_output()
            self.logger.info("WGCNA executed successfully.")
            return adjacency_matrix
        except Exception as e:
            self.logger.error(f"Error in WGCNA Network Construction: {e}")
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

        # Initialize a Series to track valid samples
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
            omics_data_clean = pd.concat([sample_ids_pheno.reset_index(drop=True), aligned_data], axis=1)
            omics_data_clean.to_csv(omics_file, index=False)
            self.logger.info(f"Cleaned omics data saved to {omics_file}")

        # Determine which samples are valid across all omics datasets
        num_valid_samples = valid_samples.sum()
        self.logger.info(f"Number of valid samples after preprocessing: {num_valid_samples} out of {len(sample_ids_pheno)}")

        if num_valid_samples == 0:
            self.logger.error("No valid samples remaining after preprocessing. Aborting WGCNA run.")
            raise ValueError("No valid samples remaining after preprocessing.")

    def run_wgcna(self) -> None:
        """
        Executes the R script for WGCNA.

        Constructs the command to run the WGCNA R script with appropriate arguments
        and captures its output.

        Raises:
            FileNotFoundError: If the WGCNA R script is not found.
            subprocess.CalledProcessError: If the R script execution fails.
        """
        # Construct argument strings
        omics_files_str = ','.join(self.omics_list)

        # Determine the path to the R script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        r_script = os.path.join(script_dir, "WGCNA.R")

        if not os.path.isfile(r_script):
            self.logger.error(f"R script not found: {r_script}")
            raise FileNotFoundError(f"R script not found: {r_script}")

        # Construct the command to execute the R script
        command = [
            "Rscript",
            r_script,
            self.phenotype_file,           # args[1]: phenotype_file
            omics_files_str,              # args[2]: omics_files (comma-separated)
            str(self.soft_power),         # args[3]: soft_power
            str(self.min_module_size),    # args[4]: min_module_size
            str(self.merge_cut_height),   # args[5]: merge_cut_height
            self.output_dir,              # args[6]: output_dir
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
            self.logger.info("WGCNA R script executed successfully.")
            self.logger.debug(f"WGCNA Output:\n{result.stdout}")

            if result.stderr:
                self.logger.warning(f"WGCNA Warnings/Errors:\n{result.stderr}")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"R script execution failed: {e.stderr}")
            raise

    def load_global_network(self) -> pd.DataFrame:
        """
        Loads the global network adjacency matrix generated by WGCNA.

        Returns:
            pd.DataFrame: Adjacency matrix of the global network.

        Raises:
            FileNotFoundError: If the global network CSV file is not found.
            pd.errors.EmptyDataError: If the CSV file is empty.
        """
        global_network_csv = os.path.join(self.output_dir, "global_network.csv")

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

    def read_adjacency_matrix(self) -> pd.DataFrame:
        """
        Reads and returns the global network adjacency matrix.

        Returns:
            pd.DataFrame: Adjacency matrix of the global network.
        """
        return self.load_global_network()

    def cleanup_output(self) -> None:
        """
        Cleans up and reorganizes WGCNA output files.

        Moves `.RData` and `.csv` files into a dedicated
        `wgcna_results` subdirectory within the `wgcna_output` directory for better organization.

        Raises:
            Exception: If any error occurs during the cleanup process.
        """
        import shutil

        try:
            saving_dir = self.output_dir
            logger = self.logger

            # Define the target directory for organized outputs
            results_dir = os.path.join(saving_dir, "wgcna_results")
            os.makedirs(results_dir, exist_ok=True)

            # Move `.RData` and `.csv` files to `wgcna_results` directory
            for file_name in os.listdir(saving_dir):
                if file_name.endswith(".RData") or file_name.endswith(".csv"):
                    src_file = os.path.join(saving_dir, file_name)
                    shutil.move(src_file, results_dir)
                    logger.info(f"Moved {file_name} to {results_dir}")

            logger.info("Cleanup and reorganization completed successfully.")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise
