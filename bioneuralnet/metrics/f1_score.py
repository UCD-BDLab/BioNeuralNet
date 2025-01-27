import pandas as pd
from ..utils.logger import get_logger


class F1Score:
    """
    F1 Score Metric Class.

    Computes the F1 score between predictions and targets.
    Suitable for classification tasks.
    """

    def __init__(self, average: str = "binary"):
        """
        Initialize the F1 Score metric.
        """
        self.logger = get_logger(__name__)
        self.average = average

    def compute(self, predictions: pd.Series, targets: pd.Series) -> float:
        """
        Compute the F1 score between predictions and targets.
        Parameter
        Returns
        Raises
        """
        self.logger.info(f"Computing F1 Score with average='{self.average}'.")

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
        fn = ((predictions == 0) & (targets == 1)).sum()

        precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * (precision * recall) / (precision + recall)

        self.logger.info(f"F1 Score: {f1}")
        return f1
