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
    
    Attributes:

        phenotype_df (pd.DataFrame): DataFrame containing phenotype data, shape [samples x 1 or more].
        omics_dfs (List[pd.DataFrame]): List of omics DataFrames.
        data_types (List[str]): List of omics data type strings (e.g. ["Genes", "miRNA"]).
        kfold (int): Number of folds for cross-validation. Default=5.
        eval_method (str): e.g. 'accuracy', 'auc', 'f1', or 'Rsquared' (if you patch SmCCNet).
        subSampNum (int): # of subsamplings. Default=50.
        summarization (str): 'NetSHy', 'PCA', or 'SVD'. Default='NetSHy'.
        seed (int): Random seed. Default=123.
        ncomp_pls (int): # of components for PLS. 0 => no PLS. Default=0.
        between_shrinkage (float): Shrink factor for multi-omics correlation. Default=5.0.
        output_dir (str): Folder to write temp files. If None, uses a temporary directory.
    """
    def __init__(
        self,
        phenotype_df: pd.DataFrame,
        omics_dfs: List[pd.DataFrame],
        data_types: List[str],
        kfold: int = 5,
        eval_method: str = "",        
        subSampNum: int = 1000,
        summarization: str = "NetSHy",
        seed: int = 723,
        ncomp_pls: int = 0,              
        between_shrinkage: float = 5.0, 
        output_dir: str = None
    ):
        """
        Initializes the SmCCNet instance.

        Args:
            phenotype_df (pd.DataFrame): DataFrame containing phenotype data, shape [samples x 1 or more].
            omics_dfs (List[pd.DataFrame]): List of omics DataFrames.
            data_types (List[str]): List of omics data type strings (e.g. ["Genes", "miRNA"]).
            kfold (int): Number of folds for cross-validation. Default=5.
            eval_method (str): e.g. 'accuracy', 'auc', 'f1', or 'Rsquared' (if you patch SmCCNet).
            subSampNum (int): # of subsamplings. Default=50.
            summarization (str): 'NetSHy', 'PCA', or 'SVD'. Default='NetSHy'.
            seed (int): Random seed. Default=123.
            ncomp_pls (int): # of components for PLS. 0 => no PLS. Default=0.
            between_shrinkage (float): Shrink factor for multi-omics correlation. Default=5.0.
            output_dir (str): Folder to write temp files. If None, uses a temporary directory.
        """
        rscript_path = shutil.which("Rscript")
        if rscript_path is None:
            raise EnvironmentError("Rscript not found in system PATH. R is required to run SmCCNet.")
            
        self.phenotype_df = phenotype_df
        self.omics_dfs = omics_dfs
        self.data_types = data_types
        self.kfold = kfold
        self.eval_method = eval_method
        self.subSampNum = subSampNum
        self.summarization = summarization
        self.seed = seed
        self.ncomp_pls = ncomp_pls
        self.between_shrinkage = between_shrinkage

        self.logger = get_logger(__name__)
        self.logger.info("Initialized SmCCNet with parameters:")
        self.logger.info(f"K-Fold: {self.kfold}")
        self.logger.info(f"Summarization: {self.summarization}")
        self.logger.info(f"Evaluation method: {self.eval_method}")
        self.logger.info(f"ncomp_pls: {self.ncomp_pls}")
        self.logger.info(f"subSampNum: {self.subSampNum}")
        self.logger.info(f"BetweenShrinkage: {self.between_shrinkage}")
        self.logger.info(f"Seed: {self.seed}")

        if len(self.omics_dfs) != len(self.data_types):
            self.logger.error("Number of omics DataFrames does not match number of data types.")
            raise ValueError("Mismatch between omics dataframes and data types.")
        
        if eval_method in ("auc","accuracy","f1"):
            uniques = set(phenotype_df.iloc[:, 0].unique())
            if not uniques.issubset({0,1}):
                raise ValueError("eval_method=classification, but phenotype is not strictly 0/1.")
        
        if eval_method == "Rsquared" and ncomp_pls>0:
            raise ValueError("Continuous eval can't use PLS. Set ncomp_pls=0 for CCA.")

        # output directory
        if output_dir is None:
            self.temp_dir_obj = tempfile.TemporaryDirectory()
            self.output_dir = self.temp_dir_obj.name
            self.logger.info(f"No output_dir provided; using temporary directory: {self.output_dir}")
        else:
            self.output_dir = output_dir
            self.logger.info(f"Output directory set to: {self.output_dir}")
            # create the directory with pathlib
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)


    def preprocess_data(self) -> Dict[str, Any]:
        """
        Preprocess the phenotype and omics data so they either:
        - All share the exact same named index already, OR
        - We rename them all to a single, consistent index name (e.g. 'SampleID').

        Then we standardize the IDs (strip + uppercase), intersect them to ensure
        overlapping samples, and serialize each DataFrame to CSV.

        Returns:
            Dict[str, Any]: A dictionary with keys 'phenotype', 'omics_1', etc.
        """
        self.logger.info("Validating and serializing input data for SmCCNet...")

        all_index_names = {self.phenotype_df.index.name}
        for df in self.omics_dfs:
            all_index_names.add(df.index.name)

        if len(all_index_names) > 1 or None in all_index_names:
            new_index_name = "SampleID"
            self.logger.info(
                f"Indexes differ or are unnamed. Renaming all indexes to '{new_index_name}'."
            )
            self.phenotype_df.index.name = new_index_name
            for df in self.omics_dfs:
                df.index.name = new_index_name
        else:
            self.logger.info("All DataFrames already share the same named index. No rename needed.")

        self.phenotype_df.index = (
            self.phenotype_df.index.astype(str).str.strip().str.upper()
        )
        for df in self.omics_dfs:
            df.index = df.index.astype(str).str.strip().str.upper()

        common_ids = set(self.phenotype_df.index)
        for df in self.omics_dfs:
            common_ids &= set(df.index)

        if not common_ids:
            raise ValueError(
                "No overlapping sample IDs found among phenotype and omics data."
            )

        common_ids_ordered = [idx for idx in self.phenotype_df.index if idx in common_ids]
        pheno_df = self.phenotype_df.loc[common_ids_ordered]
        omics_dfs_processed = [
            df.loc[common_ids_ordered] for df in self.omics_dfs
        ]

        serialized_data = {}
        serialized_data["phenotype"] = pheno_df.to_csv(index=True)
        for i, df in enumerate(omics_dfs_processed, start=1):
            key = f"omics_{i}"
            serialized_data[key] = df.to_csv(index=True)
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

            rscript_path = shutil.which("Rscript")
            if rscript_path is None:
                raise EnvironmentError("Rscript not found in system PATH. R is required to run SmCCNet.")
            
            ncomp_pls_arg = str(self.ncomp_pls) if self.ncomp_pls != 0 else ""

            command = [
                rscript_path,
                r_script,
                ",".join(self.data_types),
                str(self.kfold),
                self.summarization,
                str(self.seed),
                self.eval_method,
                ncomp_pls_arg,
                str(self.subSampNum),
                str(self.between_shrinkage),
            ]
            self.logger.debug(f"Running command: {' '.join(command)} in cwd={self.output_dir}")

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
                self.logger.warning(f"SMCCNET R script warnings/errors:\n{result.stderr}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"R script execution failed: {e.stderr}")
            raise
        except Exception as e:
            self.logger.error(f"Error during SmCCNet execution: {e}\n")
            raise

    def get_clusters(self) -> list[pd.DataFrame, Any]:
        """
        Retrieves the subnetwork clusters generated by SmCCNet.

        Returns:
            list[pd.DataFrame, Any]: A list containing the cluster DataFrame and the cluster summary.
        """
        try:
            clusters_path = Path(self.output_dir)
            clusters_names = list(clusters_path.glob("size_*.csv"))
            clusters = []
            for cluster in clusters_names:
                #cluster_path = Path(self.output_dir / cluster)
                cluster_df = pd.read_csv(cluster, index_col=0)
                clusters.append(cluster_df)

            self.logger.info(f"Found {len(clusters)} clusters in {self.output_dir}.")
            return clusters[::-1]
        except Exception as e:
            self.logger.error(f"Error reading cluster summary: {e}")
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
            adjacency_path = Path(self.output_dir) / "GlobalNetwork.csv"
            self.logger.info(f"Reading Global Network from: {adjacency_path}")
            adjacency_df = pd.read_csv(adjacency_path, index_col=0)
            self.logger.info(f"Global Network shape: {adjacency_df.shape}")
            clusters = self.get_clusters()
            self.logger.info("GlobalNetwork stored at index 0 and clusters stored as a list of dataframes at index 1.")
            self.logger.info("SmCCNet workflow completed successfully.")
            return adjacency_df, clusters
        except Exception as e:
            self.logger.error(f"Error in SmCCNet workflow: {e}")
            raise
