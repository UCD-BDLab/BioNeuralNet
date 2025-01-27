import pandas as pd
from ..utils.logger import get_logger


class MSE:
    """
    Computes the mean squared error between predictions and targets.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    def compute(self, predictions: pd.Series, targets: pd.Series) -> float:
        """
        Compute the mean squared error between predictions and targets.
        Parameters
        Returns
        Raises
        """
        self.logger.info("Computing Mean Squared Error (MSE).")

        if predictions.empty or targets.empty:
            self.logger.error("Predictions and targets must not be empty.")
            raise ValueError("Predictions and targets must not be empty.")

        if len(predictions) != len(targets):
            self.logger.error("Predictions and targets must have the same length.")
            raise ValueError("Predictions and targets must have the same length.")

        mse = ((predictions - targets) ** 2).mean()
        self.logger.info(f"Mean Squared Error (MSE): {mse}")
        return mse
