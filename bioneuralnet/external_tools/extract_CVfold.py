import os
import shutil
import subprocess
import pandas as pd
from pathlib import Path

def load_r_export_folds(base_path: str, num_omics: int, k: int = 5) -> dict:
    """Loads the specific SmCCNet directory structure exported from R.

    This function iterates through the cross-validation fold directories (fold_1, fold_2, etc.)
    and loads the associated omics CSV files and phenotype data into NumPy arrays.

    Args:

        base_path (str): The base directory containing the 'fold_N' subdirectories.
        num_omics (int): The number of omics data blocks to load per fold.
        k (int): The number of cross-validation folds to load. Defaults to 5.

    Returns:

        dict: A dictionary where keys are fold names (e.g., 'fold_1') and values are 
        dictionaries containing 'X_train' (list of numpy arrays), 'X_test' (list of numpy arrays),
        'Y_train' (numpy array), and 'Y_test' (numpy array).

    Raises:

        FileNotFoundError: If a required fold directory or CSV file cannot be found.

    """
    folddata = {}
    print(f"Loading R-exported folds from: {base_path}")
    
    for i in range(1, k + 1):
        fold_key = f"fold_{i}"
        fold_dir = os.path.join(base_path, fold_key)
        
        if not os.path.exists(fold_dir):
            raise FileNotFoundError(f"Could not find directory: {fold_dir}")
            
        x_train_list = []
        x_test_list = []
        
        for omics_idx in range(1, num_omics + 1):
            xtrain_path = os.path.join(fold_dir, f"X_train_Omics_{omics_idx}.csv")
            xtest_path = os.path.join(fold_dir, f"X_test_Omics_{omics_idx}.csv")
            
            x_train = pd.read_csv(xtrain_path).to_numpy()
            x_test = pd.read_csv(xtest_path).to_numpy()
            
            x_train_list.append(x_train)
            x_test_list.append(x_test)
            
        ytrain_path = os.path.join(fold_dir, "Y_train.csv")
        ytest_path = os.path.join(fold_dir, "Y_test.csv")
        
        y_train = pd.read_csv(ytrain_path).iloc[:, 0].to_numpy()
        y_test = pd.read_csv(ytest_path).iloc[:, 0].to_numpy()
        
        folddata[fold_key] = {
            "X_train": x_train_list,
            "X_test": x_test_list,
            "Y_train": y_train,
            "Y_test": y_test
        }
        
    print(f"Successfully loaded {k} folds.")
    return folddata


def extract_and_load_folds(output_path: str, num_omics: int = 3, k: int = 5) -> dict:
    """Extracts .Rdata fold files into CSVs using an R script, then loads them.

    This function acts as a wrapper to execute the external 'extract_CVfold.R' script,
    which parses 'CVFold.Rdata' and 'globalNetwork.Rdata' into a standard directory
    structure of CSVs. Once the R script completes successfully, it loads the data
    into memory using `load_r_export_folds`.

    Args:

        output_path (str): The target directory containing the source .Rdata files.
        num_omics (int): The number of omics data blocks to process. Defaults to 3.
        k (int): The number of cross-validation folds. Defaults to 5.

    Returns:

        dict: A dictionary containing the parsed cross-validation fold data.

    Raises:

        EnvironmentError: If 'Rscript' is not found in the system path.
        FileNotFoundError: If the required 'extract_CVfold.R' script is missing.
        RuntimeError: If the R script execution fails and returns a non-zero exit code.

    """
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise EnvironmentError("Rscript not found in system path.")

    target_dir = Path(output_path).resolve()
    
    script_path = (Path(__file__).parent / "extract_CVfold.R").resolve()

    if not script_path.exists():
        raise FileNotFoundError(f"Missing required R script: {script_path}")

    cmd = [rscript, str(script_path), str(target_dir)]
    print(f"Running Rscript command: {' '.join(cmd)}")

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.stdout:
        print(f"Rscript stdout:\n{proc.stdout}")
        
    if proc.stderr:
        if proc.returncode == 0:
            print(f"Rscript messages:\n{proc.stderr}")
        else:
            print(f"Rscript stderr:\n{proc.stderr}")

    if proc.returncode != 0:
        raise RuntimeError(f"R conversion failed with return code: {proc.returncode}")

    export_base_dir = os.path.join(str(target_dir), "CV_Export")

    return load_r_export_folds(base_path=export_base_dir, num_omics=num_omics, k=k)