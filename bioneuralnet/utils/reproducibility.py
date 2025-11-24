import os
import random
import torch
import numpy as np
from .logger import get_logger

logger = get_logger(__name__)

def set_seed(seed_value: int) -> None:
    """
    Sets seeds for maximum reproducibility across Python, NumPy, and PyTorch.

    This function sets global random seeds and configures PyTorch/CUDNN to use deterministic algorithms, ensuring that the experiment produces the exact same numerical result across different runs.

    Args:

        seed_value (int): The integer value to use as the random seed.

    Returns:

        None

    """
    logger.info(f"Setting global seed for reproducibility to: {seed_value}")

    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        logger.info("CUDA available. Applying seed to all GPU operations")
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        logger.info("CUDA not available. Seeding only CPU operations")

    logger.info("Seed setting complete")
