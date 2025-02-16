import subprocess
import shutil
from pathlib import Path
from .logger import get_logger

def rdata_to_csv_file(rdata_file: Path, csv_file: Path) -> None:
    """
    Convert an .Rdata file to CSV by invoking the rdata_to_csv.R script using Rscript.

    This function assumes that:
      - The R script '=data_to_csv.R' is in the same directory as this module.
      - R is installed and Rscript is available in the system PATH.
      - The RData file contains an object named either "AdjacencyMatrix" or "M".

    Args:
        rdata_file (Path): Path to the input .Rdata file.
        csv_file (Path): Path to the output CSV file.

    Raises:
        EnvironmentError: If Rscript is not found.
        FileNotFoundError: If the R script is not found.
        Exception: If the R script fails to execute.
    """
    logger = get_logger(__name__)
    rscript_path = shutil.which("Rscript")
    if rscript_path is None:
        raise EnvironmentError("Rscript not found in system PATH.")

    script_path = Path(__file__).parent / "rdata_to_csv.R"
    if not script_path.exists():
        raise FileNotFoundError(f"R script not found: {script_path}")

    command = [rscript_path, str(script_path), str(rdata_file), str(csv_file)]
    logger.info(f"Running command: {' '.join(command)}")

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        logger.info("Error executing R script:")
        logger.info(result.stderr)
        raise Exception("R script execution failed.")
    else:
        logger.info(result.stdout)
        logger.info(f"CSV file saved to: {csv_file}")