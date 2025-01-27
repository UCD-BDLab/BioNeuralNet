import pandas as pd
from ..utils.logger import get_logger


class Correlation:
    """
    Computes the Pearson correlation coefficient between predictions and targets.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    def compute(self, predictions: pd.Series, targets: pd.Series) -> float:
        """
        Compute the Pearson correlation coefficient between predictions and targets.
        Parameters
        Returns
        Raises
        """
        self.logger.info("Computing Pearson correlation coefficient.")

        if predictions.empty or targets.empty:
            self.logger.error("Predictions and targets must not be empty.")
            raise ValueError("Predictions and targets must not be empty.")

        if len(predictions) != len(targets):
            self.logger.error("Predictions and targets must have the same length.")
            raise ValueError("Predictions and targets must have the same length.")

        correlation = predictions.corr(targets)
        self.logger.info(f"Pearson correlation coefficient: {correlation}")
        return correlation if not pd.isna(correlation) else 0.0
