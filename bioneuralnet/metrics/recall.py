import pandas as pd
from ..utils.logger import get_logger


class Recall:
    """
    Computes the recall score between predictions and targets.
    for classification tasks.
    """

    def __init__(self, average: str = "binary"):
        """
        Initialize the Recall metric.
        """
        self.logger = get_logger(__name__)
        self.average = average

    def compute(self, predictions: pd.Series, targets: pd.Series) -> float:
        """
        Compute the recall score between predictions and targets.
        Parameters
        Returns
        Raises
        """
        self.logger.info(f"Computing Recall with average='{self.average}'.")

        if predictions.empty or targets.empty:
            self.logger.error("Predictions and targets must not be empty.")
            raise ValueError("Predictions and targets must not be empty.")

        if len(predictions) != len(targets):
            self.logger.error("Predictions and targets must have the same length.")
            raise ValueError("Predictions and targets must have the same length.")

        if self.average != "binary":
            self.logger.error(
                "Only 'binary' average is supported without scikit-learn."
            )
            raise NotImplementedError("Only 'binary' average is supported.")

        tp = ((predictions == 1) & (targets == 1)).sum()
        fn = ((predictions == 0) & (targets == 1)).sum()

        recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        self.logger.info(f"Recall Score: {recall}")
        return recall
