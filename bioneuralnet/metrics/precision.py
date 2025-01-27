import pandas as pd
from ..utils.logger import get_logger


class Precision:
    """
    Computes the precision score between predictions and targets.
    for classification tasks.
    """

    def __init__(self, average: str = "binary"):
        """
        Initialize the Precision metric.
        """
        self.logger = get_logger(__name__)
        self.average = average

    def compute(self, predictions: pd.Series, targets: pd.Series) -> float:
        """
        Compute the precision score between predictions and targets.
        Parameters
        Returns
        Raises
        """
        self.logger.info(f"Computing Precision with average='{self.average}'.")

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
        fp = ((predictions == 1) & (targets == 0)).sum()

        precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        self.logger.info(f"Precision Score: {precision}")
        return precision
