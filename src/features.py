"""Domain-specific Concrete Science Feature Engineering."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ConcreteFeatureEngineering(BaseEstimator, TransformerMixin):
    """Engineers domain-informed features based on concrete mix laws (e.g. Abrams' Law).

    Features generated:
        - total_binder: Cement + Blast Furnace Slag + Fly Ash
        - water_cement_ratio: Water / Cement (Abrams' law)
        - water_binder_ratio: Water / Total Binder (Concrete Technology fundamental)
        - aggregate_ratio: Coarse Aggregate / Fine Aggregate
        - sp_binder_ratio: Superplasticizer / Binder ratio
        - log_age: np.log1p(Age) (curing strength develops logarithmically)
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_mat = np.asarray(X)
        if X_mat.shape[1] < 8:
            raise ValueError(f"Expected at least 8 features, got {X_mat.shape[1]}")

        cement = X_mat[:, 0]
        slag = X_mat[:, 1]
        fly_ash = X_mat[:, 2]
        water = X_mat[:, 3]
        sp = X_mat[:, 4]
        coarse = X_mat[:, 5]
        fine = X_mat[:, 6]
        age = X_mat[:, 7]

        binder = cement + slag + fly_ash
        wc_ratio = water / (cement + self.eps)
        wb_ratio = water / (binder + self.eps)
        agg_ratio = coarse / (fine + self.eps)
        sp_binder = sp / (binder + self.eps)
        log_age = np.log1p(age)

        engineered = np.column_stack([
            X_mat,
            binder,
            wc_ratio,
            wb_ratio,
            agg_ratio,
            sp_binder,
            log_age
        ])
        return engineered
