"""
Baseline models for sample-level PD vs Control classification.

This module implements non-graph models (logistic regression, MLP) that
train on sample-level expression data for comparison with GNN models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bioneuralnet.utils import get_logger

logger = get_logger(__name__)


@dataclass
class BaselineResults:
    """
    Container for baseline model evaluation results.

    Attributes
    ----------
    accuracy : float
        Classification accuracy.
    f1_score : float
        F1-score (macro-averaged).
    f1_weighted : float
        Weighted F1-score.
    confusion_matrix : np.ndarray
        Confusion matrix (2x2 for binary classification).
    classification_report : str
        Detailed classification report.
    """

    accuracy: float
    f1_score: float
    f1_weighted: float
    confusion_matrix: np.ndarray
    classification_report: str


class BaselineModel:
    """
    Baseline model for sample-level PD vs Control classification.

    Supports logistic regression and multi-layer perceptron (MLP) models.
    Trains on sample-level expression data (not graph-based).
    """

    def __init__(
        self,
        model_type: str = "logistic",
        random_state: int = 42,
        test_size: float = 0.2,
        **model_kwargs,
    ):
        """
        Initialize baseline model.

        Parameters
        ----------
        model_type : str, default="logistic"
            Model type: "logistic" or "mlp".
        random_state : int, default=42
            Random seed for reproducibility.
        test_size : float, default=0.2
            Fraction of data to use for testing.
        **model_kwargs
            Additional keyword arguments passed to the model.
        """
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.test_size = test_size
        self.model_kwargs = model_kwargs

        if self.model_type == "logistic":
            self.model = LogisticRegression(
                random_state=random_state,
                max_iter=1000000,
                **model_kwargs,
            )
        elif self.model_type == "mlp":
            self.model = MLPClassifier(
                random_state=random_state,
                max_iter=1000000,
                **model_kwargs,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'logistic' or 'mlp'.")

        self.scaler = StandardScaler()
        self.is_fitted = False

        logger.info(
            f"Initialized {model_type} baseline model with random_state={random_state}."
        )

    def fit(
        self,
        expression_df: pd.DataFrame,
        sample_metadata: pd.DataFrame,
        condition_col: str = "condition",
    ) -> BaselineModel:
        """
        Train the baseline model on expression data.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Gene expression matrix (genes × samples).
        sample_metadata : pd.DataFrame
            Sample metadata with condition labels.
        condition_col : str, default="condition"
            Column name in sample_metadata containing condition labels.

        Returns
        -------
        BaselineModel
            Self (for method chaining).
        """
        logger.info("=" * 60)
        logger.info("Training baseline model.")
        logger.info("=" * 60)

        # Prepare data: transpose to samples × genes
        X = expression_df.T  # samples × genes
        y = sample_metadata[condition_col]

        # Align X and y
        common_samples = list(set(X.index) & set(y.index))
        X = X.loc[common_samples]
        y = y.loc[common_samples]

        # Encode labels: PD=1, CC=0
        y_encoded = (y == "PD").astype(int)

        logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {y_encoded.value_counts().to_dict()}")

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_encoded,
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"Test F1-score (macro): {f1:.4f}")
        logger.info(f"Test F1-score (weighted): {f1_weighted:.4f}")

        self.is_fitted = True
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred

        logger.info("=" * 60)
        logger.info("Baseline model training complete!")
        logger.info("=" * 60)

        return self

    def evaluate(self) -> BaselineResults:
        """
        Evaluate the trained model and return metrics.

        Returns
        -------
        BaselineResults
            Evaluation results including accuracy, F1-score, confusion matrix.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation.")

        accuracy = accuracy_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred, average="macro")
        f1_weighted = f1_score(self.y_test, self.y_pred, average="weighted")
        cm = confusion_matrix(self.y_test, self.y_pred)
        report = classification_report(
            self.y_test, self.y_pred, target_names=["Control", "PD"]
        )

        return BaselineResults(
            accuracy=accuracy,
            f1_score=f1,
            f1_weighted=f1_weighted,
            confusion_matrix=cm,
            classification_report=report,
        )

    def predict(self, expression_df: pd.DataFrame) -> np.ndarray:
        """
        Predict labels for new samples.

        Parameters
        ----------
        expression_df : pd.DataFrame
            Gene expression matrix (genes × samples).

        Returns
        -------
        np.ndarray
            Predicted labels (0=Control, 1=PD).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        X = expression_df.T  # samples × genes
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


def train_baseline(
    expression_df: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    model_type: str = "logistic",
    condition_col: str = "condition",
    random_state: int = 42,
    test_size: float = 0.2,
    **model_kwargs,
) -> Tuple[BaselineModel, BaselineResults]:
    """
    Convenience function to train and evaluate a baseline model.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Gene expression matrix (genes × samples).
    sample_metadata : pd.DataFrame
        Sample metadata with condition labels.
    model_type : str, default="logistic"
        Model type: "logistic" or "mlp".
    condition_col : str, default="condition"
        Column name in sample_metadata containing condition labels.
    random_state : int, default=42
        Random seed for reproducibility.
    test_size : float, default=0.2
        Fraction of data to use for testing.
    **model_kwargs
        Additional keyword arguments passed to the model.

    Returns
    -------
    Tuple[BaselineModel, BaselineResults]
        (trained model, evaluation results)
    """
    model = BaselineModel(
        model_type=model_type,
        random_state=random_state,
        test_size=test_size,
        **model_kwargs,
    )
    model.fit(expression_df, sample_metadata, condition_col=condition_col)
    results = model.evaluate()
    return model, results
