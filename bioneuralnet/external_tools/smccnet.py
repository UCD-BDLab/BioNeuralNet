import os
import subprocess
import pandas as pd
from pathlib import Path
import json
import tempfile
from typing import List, Dict, Any, Optional, Union
from ..utils.logger import get_logger
import shutil

import os
import sys
import json
import time
import threading
import itertools
import shutil
import subprocess
from pathlib import Path
from importlib.resources import files

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
        eval_method (str): e.g. 'accuracy', 'auc', 'f1', or 'Rsquared'.
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
        subSampNum: int = 500,
        summarization: str = "NetSHy",
        seed: int = 119,
        ncomp_pls: int = 0,
        between_shrinkage: float = 5.0,
        cut_height: float = (1.0 - 0.1**10.0),
        preprocess: int = 0,
        output_dir: Optional[Union[str, Path]] = None,
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
        self.logger = get_logger(__name__)
        self.rscript_path = shutil.which("Rscript")
        if self.rscript_path is None:
            raise EnvironmentError("Rscript not found in system PATH. R is required to run SmCCNet.")

        try:
            r_script_path = files("bioneuralnet.external_tools").joinpath("SmCCNet.R")
            if not r_script_path.is_file():
                raise FileNotFoundError

            self.r_script = str(r_script_path)
            self.logger.info(f"Using R script via importlib: {self.r_script}")

        except Exception:
            script_dir = Path(__file__).resolve().parent
            r_script_path = script_dir / "SmCCNet.R"

            if not r_script_path.is_file():
                raise FileNotFoundError(f"SmCCNet.R script not found via importlib or local path: {r_script_path}")

            self.r_script = str(r_script_path)
            self.logger.warning(f"Using fallback R script path: {self.r_script}")

        if isinstance(phenotype_df, pd.Series):
            phenotype_df = phenotype_df.to_frame(name="phenotype")

        if isinstance(phenotype_df, pd.DataFrame) and phenotype_df.shape[1] > 1:
            self.logger.warning("Phenotype DataFrame has more than one column. Renaming to phenotype and keeping only the first column")
            phenotype_df = phenotype_df.iloc[:, :1]
            phenotype_df.columns = ["phenotype"]

        if not isinstance(phenotype_df, pd.DataFrame):
            raise ValueError("phenotype_df must be a pandas DataFrame or Series.")

        self.phenotype_df = phenotype_df.copy(deep=True)

        self.omics_dfs = []
        for df in omics_dfs:
            self.omics_dfs.append(df.copy(deep=True))

        self.data_types = data_types
        self.kfold = kfold
        self.eval_method = eval_method
        self.subSampNum = subSampNum
        self.summarization = summarization
        self.seed = seed
        self.ncomp_pls = ncomp_pls
        self.between_shrinkage = between_shrinkage
        self.cut_height = cut_height
        self.preprocess = preprocess


        self.logger = get_logger(__name__)
        self.logger.info("Initialized SmCCNet with parameters:")
        self.logger.info(f"K-Fold: {self.kfold}")
        self.logger.info(f"Summarization: {self.summarization}")
        self.logger.info(f"Evaluation method: {self.eval_method}")
        self.logger.info(f"ncomp_pls: {self.ncomp_pls}")
        self.logger.info(f"subSampNum: {self.subSampNum}")
        self.logger.info(f"BetweenShrinkage: {self.between_shrinkage}")
        self.logger.info(f"Seed: {self.seed}")
        self.logger.info(f"Cut height: {self.cut_height}")

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
            self.output_dir = Path(self.temp_dir_obj.name)
            self.logger.info(f"No output_dir provided; using temporary directory: {self.output_dir}")
        else:
            self.output_dir = Path(output_dir)
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
        base_index = self.phenotype_df.index.astype(str).str.strip().str.upper()

        for i, df in enumerate(self.omics_dfs, start=1):
            index_match = False
            other_index = df.index.astype(str).str.strip().str.upper()
            if not base_index.equals(other_index):
                self.logger.warning(f"Index mismatch: phenotype vs {i}.")
            else:
                index_match = True
                self.logger.info(f"Index match: phenotype & {i}.")

        self.logger.info("Validating and serializing input data for SmCCNet...")
        if self.phenotype_df.columns[0] != "phenotype":
            self.logger.warning("Renaming target column to 'phenotype' for consistency.")
            self.phenotype_df.columns = ["phenotype"]


        # if index_match == True:
        #     self.logger.info("All DataFrames already share the same named index. No rename needed.")
        # else:
        self.logger.info("Renaming indexes to a consistent name.")
        all_index_names = {self.phenotype_df.index.name}
        for df in self.omics_dfs:
            all_index_names.add(df.index.name)

        if len(all_index_names) > 1 or None in all_index_names:
            new_index_name = "SampleID"

            self.phenotype_df.index.name = new_index_name
            for df in self.omics_dfs:
                df.index.name = new_index_name

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
        self.logger.info(f"Serialized phenotype with {len(pheno_df)} samples.")

        return serialized_data


    def run_smccnet(self, serialized_data: Dict[str, Any]) -> None:
        """
        Executes the SmCCNet R script in the specified output directory,
        printing a simple spinner while it runs.
        """
        try:
            self.logger.info("Executing SmCCNet R script...")
            json_data = json.dumps(serialized_data) + "\n"

            # script_dir = os.path.dirname(os.path.abspath(__file__))

            # r_script = os.path.join(script_dir, "SmCCNet.R")
            # if not os.path.isfile(r_script):
            #     raise FileNotFoundError(f"R script not found: {r_script}")

            # rscript_path = shutil.which("Rscript")
            # if rscript_path is None:
            #     raise EnvironmentError("Rscript not found in system PATH.")
            cmd = [
                self.rscript_path,
                self.r_script,
                ",".join(self.data_types),
                str(self.kfold),
                self.summarization,
                str(self.seed),
                self.eval_method,
                str(self.ncomp_pls),
                str(self.subSampNum),
                str(self.between_shrinkage),
                str(self.cut_height),
                str(self.preprocess),
            ]
            self.logger.info(f"Running command: {cmd}")

            # fire off spinner thread
            stop_spinner = threading.Event()
            def spinner():
                for ch in itertools.cycle("|/-\\"):
                    if stop_spinner.is_set():
                        break
                    sys.stdout.write(f"\rRunning SmCCNet… {ch}")
                    sys.stdout.flush()
                    time.sleep(0.1)
                sys.stdout.write("\rSmCCNet finished.    \n")

            spin_thread = threading.Thread(target=spinner)
            spin_thread.start()
            cmd_clean: list[str] = []
            for c in cmd:
                if c is None:
                    raise ValueError("Command argument cannot be None")
                cmd_clean.append(str(c))

            # run Rscript (blocks until done)
            result = subprocess.run(
                cmd_clean,
                input=json_data,
                text=True,
                capture_output=True,
                check=True,
                cwd=self.output_dir,
            )

            # stop spinner
            stop_spinner.set()
            spin_thread.join()

            # log R output
            if result.stdout.strip():
                self.logger.info(f"SMCCNET R script output:\n{result.stdout}")
            if result.stderr.strip():
                self.logger.warning(f"SMCCNET R script warnings/errors:\n{result.stderr}")

        except subprocess.CalledProcessError as e:
            stop_spinner.set()
            spin_thread.join()
            self.logger.error(f"R script execution failed: {e.stderr}")
            raise
        except Exception as e:
            stop_spinner.set()
            spin_thread.join()
            self.logger.error(f"Error during SmCCNet execution: {e}")
            raise


    def get_clusters(self) -> list[Any]:
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
