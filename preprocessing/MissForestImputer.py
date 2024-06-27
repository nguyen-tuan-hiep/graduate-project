import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class MissForest:
    def __init__(self, max_iter=10, decreasing=True):
        self.max_iter = max_iter
        self.decreasing = decreasing

    def _initial_imputation(self, X, categorical_cols):
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                if col in categorical_cols:
                    mode = X[col].mode()[0]
                    X[col].fillna(mode, inplace=True)
                else:
                    mean = X[col].mean()
                    X[col].fillna(mean, inplace=True)
        return X

    def fit_transform(self, X, categorical_cols=[]):
        X = X.copy()

        # Print columns containing strings (categorical columns)
        str_cols = X.select_dtypes(include=['object']).columns

        X = self._initial_imputation(X, categorical_cols)
        n, p = X.shape

        missing_mask = X.isnull()
        prev_X = X.copy()
        gamma = np.inf

        for iteration in range(self.max_iter):
            for col in X.columns:
                mis = missing_mask[col]
                if mis.sum() == 0:
                    continue

                ymis = X.loc[mis, col]
                yobs = X.loc[~mis, col]
                Xmis = X.loc[mis, X.columns != col]
                Xobs = X.loc[~mis, X.columns != col]

                if col in categorical_cols:
                    model = RandomForestClassifier()
                else:
                    model = RandomForestRegressor()

                model.fit(Xobs, yobs)
                X.loc[mis, col] = model.predict(Xmis)

            # Compute the stopping criterion only for numerical columns
            num_cols = X.columns.difference(categorical_cols)
            gamma_new = ((prev_X[num_cols].astype(float) - X[num_cols].astype(float)) ** 2).sum().sum() / n / len(num_cols)
            if gamma_new > gamma:
                break

            gamma = gamma_new
            prev_X = X.copy()

        return X
