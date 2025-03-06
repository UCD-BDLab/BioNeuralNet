import subprocess
import shutil
from pathlib import Path
import pandas as pd
from .logger import get_logger

def rdata_to_df(rdata_file: Path, csv_file: Path, Object=None) -> None:
    """
    Convert an .Rdata file to CSV by invoking the rdata_to_csv.R script using Rscript.

    Parameters:

        rdata_file (Path): Path to the input .Rdata file.
        Object (str, optional): The name of the R object to load (if not default).
    
    Returns:
    
        pd.DataFrame: DataFrame loaded from the CSV converted from the RData file.
    """
    logger = get_logger(__name__)
    rscript_path = shutil.which("Rscript")
    if rscript_path is None:
        raise EnvironmentError("Rscript not found in system PATH. R is required to convert .Rdata files to CSV.")

    script_path = Path(__file__).parent / "rdata_to_df.R"
    if not script_path.exists():
        raise FileNotFoundError(f"R script not found: {script_path}")

    command = [rscript_path, str(script_path), str(rdata_file), str(csv_file), str(Object)]
    logger.info(f"Running command: {' '.join(command)}")

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        logger.info("Error executing R script:")
        logger.info(result.stderr)
        raise Exception("R script execution failed.")
    else:
        logger.info(result.stdout)
        logger.info(f"CSV file saved to: {csv_file}")

    data = pd.read_csv(csv_file)

    return data