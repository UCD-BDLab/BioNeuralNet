import logging
import os

def get_logger(name: str) -> logging.Logger:
    """Retrieves a global logger configured to write to 'bioneuralnet.log'.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_file = os.path.join(project_root, "bioneuralnet.log")

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(file_formatter)

        console_formatter = logging.Formatter("%(message)s")
        ch.setFormatter(console_formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
