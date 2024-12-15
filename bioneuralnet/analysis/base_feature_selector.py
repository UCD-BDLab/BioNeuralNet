import os
from datetime import datetime
from typing import Optional, List

import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

from ..utils.logger import get_logger


class BaseFeatureSelector:
    """
    BaseFeatureSelector Class for Shared Functionalities in Feature Selection Components.

    This abstract base class provides common methods for feature selection,
    including statistical and machine learning-based methods. It serves as a foundation
    for specialized feature selection classes like FeatureSelector and EnvironmentalExposureSelector.
    """

    def __init__(
        self,
        num_features: int = 10,
        selection_method: str = 'correlation',
        output_dir: Optional[str] = None,
    ):
        """
        Initializes the BaseFeatureSelector instance.

        Args:
            num_features (int, optional): Number of top features to select. Defaults to 10.
            selection_method (str, optional): Feature selection method ('correlation', 'lasso', 'random_forest'). Defaults to 'correlation'.
            output_dir (str, optional): Directory to save selected features. If None, creates a unique directory.
        """
        self.num_features = num_features
        self.selection_method = selection_method
        self.output_dir = output_dir if output_dir else self._create_output_dir()
        self.logger = get_logger(__name__)
        self.logger.info("Initialized BaseFeatureSelector.")

    def _create_output_dir(self) -> str:
        """
        Creates a unique output directory for the FeatureSelector run.

        Returns:
            str: Path to the created output directory.
        """
        base_dir = "feature_selection_output"
        timestamp = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
        output_dir = f"{base_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Created output directory: {output_dir}")
        return output_dir

    def perform_feature_selection(
        self,
        data: pd.DataFrame,
        phenotype: pd.Series
    ) -> pd.DataFrame:
        """
        Performs feature selection on the provided data based on the selected method.

        Args:
            data (pd.DataFrame): Data on which feature selection is to be performed.
            phenotype (pd.Series): Target phenotype data.

        Returns:
            pd.DataFrame: DataFrame containing the selected features.

        Raises:
            ValueError: If an unsupported feature selection method is specified.
        """
        self.logger.info(f"Performing feature selection using method: {self.selection_method}")

        if self.selection_method == 'correlation':
            selected_features = self._correlation_based_selection(data, phenotype)
        elif self.selection_method == 'lasso':
            selected_features = self._lasso_based_selection(data, phenotype)
        elif self.selection_method == 'random_forest':
            selected_features = self._random_forest_based_selection(data, phenotype)
        else:
            self.logger.error(f"Unsupported feature selection method: {self.selection_method}")
            raise ValueError(f"Unsupported feature selection method: {self.selection_method}")

        return selected_features

    def _correlation_based_selection(self, data: pd.DataFrame, phenotype: pd.Series) -> pd.DataFrame:
        """
        Selects top features based on correlation with phenotype.

        Args:
            data (pd.DataFrame): Data on which feature selection is to be performed.
            phenotype (pd.Series): Target phenotype data.

        Returns:
            pd.DataFrame: DataFrame containing the selected features.
        """
        self.logger.info("Performing correlation-based feature selection using ANOVA (f_classif).")
        selector = SelectKBest(score_func=f_classif, k=self.num_features)
        selector.fit(data, phenotype)
        selected_mask = selector.get_support()
        selected_features = data.columns[selected_mask]
        self.logger.info(f"Selected {len(selected_features)} features based on correlation.")
        return data[selected_features]

    def _lasso_based_selection(self, data: pd.DataFrame, phenotype: pd.Series) -> pd.DataFrame:
        """
        Selects top features based on LASSO regression coefficients.

        Args:
            data (pd.DataFrame): Data on which feature selection is to be performed.
            phenotype (pd.Series): Target phenotype data.

        Returns:
            pd.DataFrame: DataFrame containing the selected features.
        """
        self.logger.info("Performing LASSO-based feature selection.")
        lasso = LassoCV(cv=5, random_state=0).fit(data, phenotype)
        coef = pd.Series(lasso.coef_, index=data.columns)
        selected_features = coef.abs().sort_values(ascending=False).head(self.num_features).index
        self.logger.info(f"Selected {len(selected_features)} features based on LASSO coefficients.")
        return data[selected_features]

    def _random_forest_based_selection(self, data: pd.DataFrame, phenotype: pd.Series) -> pd.DataFrame:
        """
        Selects top features based on Random Forest feature importances.

        Args:
            data (pd.DataFrame): Data on which feature selection is to be performed.
            phenotype (pd.Series): Target phenotype data.

        Returns:
            pd.DataFrame: DataFrame containing the selected features.
        """
        self.logger.info("Performing Random Forest-based feature selection.")
        rf = RandomForestClassifier(n_estimators=100, random_state=0)
        rf.fit(data, phenotype)
        importances = pd.Series(rf.feature_importances_, index=data.columns)
        selected_features = importances.sort_values(ascending=False).head(self.num_features).index
        self.logger.info(f"Selected {len(selected_features)} features based on Random Forest importances.")
        return data[selected_features]

    def save_selected_features(self, selected_features: pd.DataFrame, filename: str = "selected_features.csv"):
        """
        Saves the selected features to the output directory.

        Args:
            selected_features (pd.DataFrame): DataFrame containing the selected features.
            filename (str, optional): Name of the file to save the selected features. Defaults to "selected_features.csv".
        """
        try:
            selected_features_file = os.path.join(self.output_dir, filename)
            selected_features.to_csv(selected_features_file)
            self.logger.info(f"Selected features saved to {selected_features_file}")
        except Exception as e:
            self.logger.error(f"Failed to save selected features: {e}")
            raise
