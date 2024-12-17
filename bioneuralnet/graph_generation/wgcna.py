import os
import subprocess
import pandas as pd
from typing import List
from ..utils.logger import get_logger
from datetime import datetime

class WGCNA:
    """
    WGCNA Class for Graph Construction using Weighted Gene Co-expression Network Analysis (WGCNA).

    Similar in approach to SmCCNet: no cleanup step if not required.
    """

    def __init__(
        self,
        phenotype_file: str,
        omics_list: List[str],
        data_types: List[str],
        soft_power: int = 6,
        min_module_size: int = 30,
        merge_cut_height: float = 0.25,
    ):
        # Assign parameters
        self.phenotype_file = phenotype_file
        self.omics_list = omics_list
        self.data_types = data_types
        self.soft_power = soft_power
        self.min_module_size = min_module_size
        self.merge_cut_height = merge_cut_height

        # Initialize logger
        self.logger = get_logger(__name__)
        self.logger.info("Initialized WGCNA with the following parameters:")
        self.logger.info(f"Phenotype File: {self.phenotype_file}")
        self.logger.info(f"Omics Files: {self.omics_list}")
        self.logger.info(f"Data Types: {self.data_types}")
        self.logger.info(f"Soft Power: {self.soft_power}")
        self.logger.info(f"Minimum Module Size: {self.min_module_size}")
        self.logger.info(f"Merge Cut Height: {self.merge_cut_height}")

    def _create_output_dir(self) -> str:
        """
        Creates a unique output directory for the current WGCNA run.
        """
        base_dir = "wgcna_output"
        timestamp = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
        output_dir = f"{base_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Created output directory: {output_dir}")
        return output_dir

    def run(self) -> pd.DataFrame:
        """
        Execute WGCNA to build a weighted correlation network.

        Steps:
        1. **Preprocessing**: Cleans and formats the omics data.
        2. **WGCNA Execution**: Runs the R-based WGCNA pipeline to construct a weighted correlation network.
        3. **Load Adjacency Matrix**: Reads the resulting adjacency matrix from disk.

        Returns:
            pd.DataFrame:
                A DataFrame representing the weighted correlation network adjacency matrix. Each cell value 
                in the matrix corresponds to the correlation-based edge weight between two omics features.

        Raises:
            FileNotFoundError: If required data or output files are missing.
            subprocess.CalledProcessError: If the underlying WGCNA R script fails.
            Exception: For any other unexpected issues.

        Notes:
            Ensure that WGCNA is properly installed in your R environment.
            The omics data should contain no missing values and should be normalized or standardized as required.
        """
        try:
            self.logger.info("Starting WGCNA Network Construction")
            output_dir = self._create_output_dir()
            self.preprocess_data()
            self.run_wgcna(output_dir)
            adjacency_matrix = self.load_global_network(output_dir)
            self.logger.info("WGCNA executed successfully.")
            return adjacency_matrix
        except Exception as e:
            self.logger.error(f"Error in WGCNA Network Construction: {e}")
            raise


    def preprocess_data(self) -> None:
        """
        Preprocesses the omics data similarly to SmCCNet.
        """
        self.logger.info("Preprocessing omics data for NaN or infinite values.")

        # Load phenotype data
        try:
            phenotype_data = pd.read_csv(self.phenotype_file, header=0)
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

        sample_ids_pheno = phenotype_data.iloc[:, 0]
        self.logger.info(f"Number of samples in phenotype data: {len(sample_ids_pheno)}")

        valid_samples = pd.Series([True] * len(sample_ids_pheno), index=sample_ids_pheno.index)

        for omics_file in self.omics_list:
            self.logger.info(f"Processing omics file: {omics_file}")

            try:
                omics_data = pd.read_csv(omics_file, header=0)
                self.logger.info(f"Omics data loaded with shape {omics_data.shape}")
            except FileNotFoundError:
                self.logger.error(f"Omics data file not found: {omics_file}")
                raise
            except pd.errors.EmptyDataError:
                self.logger.error(f"Omics data file is empty: {omics_file}")
                raise

            sample_ids_omics = omics_data.iloc[:, 0]
            omics_values = omics_data.iloc[:, 1:]

            aligned_data = omics_values.set_index(sample_ids_omics).loc[sample_ids_pheno].reset_index(drop=True)

            if aligned_data.isnull().values.any():
                self.logger.warning(f"NaN values detected in omics data after alignment for file: {omics_file}")
                valid_samples &= ~aligned_data.isnull().any(axis=1)

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

            omics_data_clean = pd.concat([sample_ids_pheno.reset_index(drop=True), aligned_data], axis=1)
            omics_data_clean.to_csv(omics_file, index=False)
            self.logger.info(f"Cleaned omics data saved to {omics_file}")

        num_valid_samples = valid_samples.sum()
        self.logger.info(f"Number of valid samples after preprocessing: {num_valid_samples} out of {len(sample_ids_pheno)}")

        if num_valid_samples == 0:
            self.logger.error("No valid samples remaining after preprocessing. Aborting WGCNA run.")
            raise ValueError("No valid samples remaining after preprocessing.")

    def run_wgcna(self, output_dir: str) -> None:
        """
        Executes the WGCNA R script with required arguments.
        """
        omics_files_str = ','.join(self.omics_list)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        r_script = os.path.join(script_dir, "WGCNA.R")

        if not os.path.isfile(r_script):
            self.logger.error(f"R script not found: {r_script}")
            raise FileNotFoundError(f"R script not found: {r_script}")

        command = [
            "Rscript",
            r_script,
            self.phenotype_file,          
            omics_files_str,             
            str(self.soft_power),        
            str(self.min_module_size),    
            str(self.merge_cut_height),  
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

    def load_global_network(self, output_dir: str) -> pd.DataFrame:
        """
        Loads the global network adjacency matrix generated by WGCNA.
        """
        global_network_csv = os.path.join(output_dir, "global_network.csv")

        if not os.path.isfile(global_network_csv):
            self.logger.error(f"Global network file not found: {global_network_csv}")
            raise FileNotFoundError(f"Global network file not found: {global_network_csv}")

        try:
            adjacency_matrix = pd.read_csv(global_network_csv, index_col=0)
            self.logger.info("Global network adjacency matrix loaded successfully.")
            return adjacency_matrix
        except pd.errors.EmptyDataError:
            self.logger.error(f"Global network CSV file is empty: {global_network_csv}")
            raise


