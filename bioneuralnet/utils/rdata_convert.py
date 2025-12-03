import subprocess
import shutil
from pathlib import Path
import pandas as pd
from .logger import get_logger

logger = get_logger(__name__)

def rdata_to_df(rdata_file: Path, csv_file: Path, Object=None) -> pd.DataFrame:
    """Converts an RData file to a pandas DataFrame.

    This function executes an external R script to load the .RData file, identify a suitable matrix, data frame, or graph object (e.g., igraph), and export it to CSV.

    Args:

        rdata_file (Path): Path to the input .RData file.
        csv_file (Path): Path where the temporary CSV file should be written.
        Object (str | None): Optional name of the specific object to extract; if None, the script attempts to auto-detect the first suitable object.

    Returns:

        pd.DataFrame: The converted data loaded into a pandas DataFrame.

    Raises:

        EnvironmentError: If 'Rscript' is not found in the system path.
        RuntimeError: If the R script execution fails.
        FileNotFoundError: If the resulting CSV file cannot be found.

    """
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise EnvironmentError("Rscript not found…")

    rdata_file = rdata_file.resolve()
    csv_file = csv_file.resolve()
    script = (Path(__file__).parent / "rdata_to_df.R").resolve()

    cmd = [rscript, str(script), str(rdata_file), str(csv_file), str(Object or "")]
    logger.info(f"Running Rscript command: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.stdout:
        logger.info(f"Rscript stdout:\n{proc.stdout}")
    if proc.stderr:
        logger.warning(f"Rscript stderr:\n{proc.stderr}")

    if proc.returncode != 0:
        raise RuntimeError(f"R conversion failed: {proc.returncode})")

    if not csv_file.exists():
        possibilities = [
            Path.cwd() / csv_file.name,
            script.parent / csv_file.name
        ]
        found = None
        for p in possibilities:
            if p.exists():
                found = p
                logger.warning(f"Found CSV at {p} instead of {csv_file}")
                csv_file = p
                break
        if not found:
            raise FileNotFoundError(f"No CSV at {csv_file}, nor in {possibilities}")

    return pd.read_csv(csv_file, index_col=0)
