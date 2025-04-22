import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

def evaluate_rf(X, y, n=150, mode="classification", runs=100, seed=119, return_all=False):
    """
    Evaluate Random Forest performance over multiple runs.
    """
    X = X.copy()
    y = y.copy()
    scores = []
    for run in range(runs):
        stratify = y if mode == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed + run, stratify=stratify
        )
        if mode == "regression":
            model = RandomForestRegressor(n_estimators=n, random_state=seed + run)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
        elif mode == "classification":
            model = RandomForestClassifier(n_estimators=n, random_state=seed + run)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
        else:
            raise ValueError("Mode must be 'regression' or 'classification'.")
        scores.append(score)
    if return_all:
        return scores
    else:
        return np.mean(scores), np.std(scores)
    