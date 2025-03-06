
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split

def evaluate_rf(X, y, n=50, mode="regression", seed=119):
    """
    Evaluate a random forest model on the given data.

    Parameters:

        X: feature matrix
        y: target vector
        n: number of trees in the forest
    

    returns:

        r2: R-squared for regression
        acc: accuracy for classification
        
    """

    if mode == "regression":
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
        rf = RandomForestRegressor(n_estimators=n)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        return r2
    
    elif mode == "classification":
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
        rf = RandomForestClassifier(n_estimators=n)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return acc
    
    else:
        raise ValueError("Invalid mode: please set mode for `regression` or `classification`.")

