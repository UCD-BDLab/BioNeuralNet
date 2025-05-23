import numpy as np
from typing import Union, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, f1_score
from sklearn.exceptions import DataConversionWarning
from bioneuralnet.utils import get_logger

import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
logger = get_logger(__name__)

def evaluate_model(X: np.ndarray,y: np.ndarray,model_type: str = "rf_classif",n_estimators: int = 150,runs: int = 10,seed: int = 119,) -> Tuple[
       Tuple[float, float],
       Tuple[Optional[float], Optional[float]],
       Tuple[Optional[float], Optional[float]]
   ]:
    """
    Evaluate a RandomForest model (classifier or regressor) over multiple train/test splits.

    Parameters:

        X: Feature matrix (NumPy array).
        y: Target array (NumPy array).
        model_type: "rf_classif" for classification or "rf_reg" for regression.
        n_estimators: Number of trees in the forest.
        runs: Number of train/test runs to average over.
        seed: Random seed for reproducibility.

    Returns:

    For classification, a tuple of 3 metric tuples:

        (accuracy_mean, accuracy_std)
        (f1_weighted_mean, f1_weighted_std)
        (f1_macro_mean, f1_macro_std)

    For regression, a single tuple:

        (r2_mean, r2_std)
        (None, None)
        (None, None)
    """
    X = X.copy()
    y = y.copy()

    accs, f1ws, f1ms, rsqs = [], [], [], []
    is_classif = "classif" in model_type

    for run in range(runs):
        stratify = y if is_classif else None
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed + run, stratify=stratify)

        if model_type == "rf_classif":
            mdl = RandomForestClassifier(n_estimators=n_estimators, random_state=seed + run)
        elif model_type == "rf_reg":
            mdl = RandomForestRegressor(n_estimators=n_estimators, random_state=seed + run)
        else:
            raise ValueError("model_type must be one of: rf_classif, rf_reg")

        mdl.fit(X_tr, y_tr)
        y_pred = mdl.predict(X_te)

        if is_classif:
            accs.append(accuracy_score(y_te, y_pred))
            f1ws.append(f1_score(y_te, y_pred, average="weighted"))
            f1ms.append(f1_score(y_te, y_pred, average="macro"))
        else:
            rsqs.append(r2_score(y_te, y_pred))

    if is_classif:
        return (
            (np.mean(accs), np.std(accs)),
            (np.mean(f1ws), np.std(f1ws)),
            (np.mean(f1ms), np.std(f1ms)))
    else:
        return (
            (np.mean(rsqs), np.std(rsqs)),
            (None, None),
            (None, None))


def evaluate_rf(X: np.ndarray,y: np.ndarray,mode: str = "classification",n_estimators: int = 150,runs: int = 5,seed: int = 119) -> Union[
    Tuple[Tuple[float, float], Tuple[Optional[float], Optional[float]], Tuple[Optional[float], Optional[float]]],
    Tuple[float, float]]:
    """
    Convenience wrapper for evaluating a RandomForest model (classifier or regressor)

    Parameters:

        X: Feature matrix
        y: Target array
        mode: Either "classification" or "regression"
        n_estimators: Number of estimators in the forest
        runs: Number of cross-validation runs
        seed: Random seed

    Returns:

        For classification:

            ((accuracy_mean, accuracy_std), (f1_weighted_mean, f1_weighted_std), (f1_macro_mean, f1_macro_std))

        For regression:

            (r2_mean, r2_std)
    """
    mt = "rf_classif" if mode == "classification" else "rf_reg"
    all_results = evaluate_model(X, y,model_type=mt,n_estimators=n_estimators,runs=runs,seed=seed)

    if mode != "classification":
        # drop the two None slots
        return all_results[0]

    return all_results


def evaluate_f1w(X: np.ndarray,y: np.ndarray,model_type: str = "rf_classif",n_estimators: int = 100,runs: int = 5,seed: int = 119) -> Tuple[float, float]:
    """
    Evaluate the weighted F1-score for a RandomForest classifier across multiple runs.

    Parameters:

        X: Feature matrix
        y: Target array
        model_type: Must be "rf_classif"
        n_estimators: Number of trees
        runs: Number of train/test runs
        seed: Random seed for reproducibility

    Returns:

        tuple of (mean_f1_weighted, std_f1_weighted)
    """
    scores = []
    for run in range(runs):
        stratify = y if "classif" in model_type else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed+run, stratify=stratify)
        if model_type == "rf_classif":
            mdl = RandomForestClassifier(n_estimators=n_estimators, random_state=seed+run)
        else:
            raise ValueError("Unsupported model_type for F1 scoring")

        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        scores.append(f1_score(y_test, y_pred, average="weighted"))

    return np.mean(scores), np.std(scores)

def evaluate_f1m(X: np.ndarray,y: np.ndarray,model_type: str = "rf_classif",n_estimators: int = 100,runs: int = 5,seed: int = 119) -> Tuple[float, float]:
    """
    Evaluate the macro F1-score for a RandomForest classifier across multiple runs

    Parameters:

        X: Feature matrix
        y: Target array
        model_type: Must be "rf_classif"
        n_estimators: Number of trees.
        runs: Number of train/test runs
        seed: Random seed for reproducibility

    Returns:

        Tuple of (mean_f1_macro, std_f1_macro)
    """
    scores = []
    for run in range(runs):
        stratify = y if "classif" in model_type else None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed+run, stratify=stratify)

        if model_type == "rf_classif":
            mdl = RandomForestClassifier(n_estimators=n_estimators, random_state=seed+run)
        else:
            raise ValueError("Unsupported model_type for F1 scoring")

        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        scores.append(f1_score(y_test, y_pred, average="macro"))

    return np.mean(scores), np.std(scores)
