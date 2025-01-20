import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from ..utils.logger import get_logger


class FeatureSelector:
    """
    FeatureSelector Class for Selecting Relevant Multi-Omics Features.

    This class provides methods for feature selection using statistical and machine learning-based approaches.
    It allows selection based on correlation, LASSO regression, or Random Forest feature importances.
    """

    def __init__(
        self,
        enhanced_omics_data: pd.DataFrame,
        phenotype_data: pd.Series,
        num_features: int = 10,
        selection_method: str = "correlation",
    ):
        """
        Initializes the FeatureSelector instance.

        Args:
            enhanced_omics_data (pd.DataFrame): Enhanced multi-omics dataset with integrated embeddings.
            phenotype_data (pd.Series): Phenotype data corresponding to the samples.
            num_features (int, optional): Number of top features to select. Defaults to 10.
            selection_method (str, optional): Feature selection method ('correlation', 'lasso', 'random_forest'). Defaults to 'correlation'.
            #output_dir (str, optional): Directory to save selected features. If None, creates a unique directory.
        """
        self.logger = get_logger(__name__)
        self.logger.info("Initialized FeatureSelector.")
        self.num_features = num_features
        self.selection_method = selection_method
        self.enhanced_omics_data = enhanced_omics_data
        self.phenotype_data = phenotype_data

    def perform_feature_selection(self) -> pd.DataFrame:
        """
        Performs feature selection on the enhanced omics data based on the selected method.

        Returns:
            pd.DataFrame: DataFrame containing the selected features.

        Raises:
            ValueError: If an unsupported feature selection method is specified.
        """
        self.logger.info(
            f"Performing feature selection using method: {self.selection_method}"
        )

        if self.selection_method == "correlation":
            selected_features = self._correlation_based_selection()
        elif self.selection_method == "lasso":
            selected_features = self._lasso_based_selection()
        elif self.selection_method == "random_forest":
            selected_features = self._random_forest_based_selection()
        else:
            self.logger.error(
                f"Unsupported feature selection method: {self.selection_method}"
            )
            raise ValueError(
                f"Unsupported feature selection method: {self.selection_method}"
            )

        return selected_features

    def _correlation_based_selection(self) -> pd.DataFrame:
        """
        Selects top features based on correlation with phenotype using ANOVA.

        Returns:
            pd.DataFrame: DataFrame containing the selected features.
        """
        self.logger.info(
            "Performing correlation-based feature selection using ANOVA (f_classif)."
        )
        selector = SelectKBest(score_func=f_classif, k=self.num_features)
        selector.fit(self.enhanced_omics_data, self.phenotype_data)
        selected_mask = selector.get_support()
        selected_features = self.enhanced_omics_data.columns[selected_mask]
        self.logger.info(
            f"Selected {len(selected_features)} features based on correlation."
        )
        return self.enhanced_omics_data[selected_features]

    def _lasso_based_selection(self) -> pd.DataFrame:
        """
        Selects top features based on LASSO regression coefficients.

        Returns:
            pd.DataFrame: DataFrame containing the selected features.
        """
        self.logger.info("Performing LASSO-based feature selection.")
        # Ensure cv does not exceed number of samples
        n_samples = len(self.enhanced_omics_data)
        cv_folds = 5
        if n_samples < cv_folds:
            cv_folds = n_samples
            self.logger.warning(
                f"Reducing cv from 5 to {cv_folds} due to insufficient samples."
            )
        if cv_folds < 2:
            raise ValueError(
                f"Number of splits {cv_folds} must be at least 2 for cross-validation."
            )

        lasso = LassoCV(cv=cv_folds, random_state=0).fit(
            self.enhanced_omics_data, self.phenotype_data
        )
        coef = pd.Series(lasso.coef_, index=self.enhanced_omics_data.columns)
        selected_features = (
            coef.abs().sort_values(ascending=False).head(self.num_features).index
        )
        self.logger.info(
            f"Selected {len(selected_features)} features based on LASSO coefficients."
        )
        return self.enhanced_omics_data[selected_features]

    def _random_forest_based_selection(self) -> pd.DataFrame:
        """
        Selects top features based on Random Forest feature importances.

        Returns:
            pd.DataFrame: DataFrame containing the selected features.
        """
        self.logger.info("Performing Random Forest-based feature selection.")
        rf = RandomForestClassifier(n_estimators=100, random_state=0)
        rf.fit(self.enhanced_omics_data, self.phenotype_data)
        importances = pd.Series(
            rf.feature_importances_, index=self.enhanced_omics_data.columns
        )
        selected_features = (
            importances.sort_values(ascending=False).head(self.num_features).index
        )
        self.logger.info(
            f"Selected {len(selected_features)} features based on Random Forest importances."
        )
        return self.enhanced_omics_data[selected_features]

    def run_feature_selection(self) -> pd.DataFrame:
        """
        Executes the feature selection process and saves the results.

        Returns:
            pd.DataFrame: DataFrame containing the selected features.
        """
        self.logger.info("Starting feature selection on enhanced omics data.")
        selected_features = self.perform_feature_selection()
        self.logger.info(
            "Feature selection on enhanced omics data completed successfully."
        )
        return selected_features
