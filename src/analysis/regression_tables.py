"""
regression_tables.py

Implements regression tasks for analyzing CNN predictions with stock returns or fundamentals.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, r2_score
import statsmodels.api as sm

from src.utils.config import LIGHT_CSV_RES_DIR
from src.data.dgp_config import CACHE_DIR
from src.utils import utilities as ut


class SMRegression:
    """
    Statsmodels-based Regression: supports logistic or OLS.
    """

    def __init__(
        self,
        regression_type: str,
        logit_threshold: float = 0.0,
        rank_norm_x: bool = False,
        ols_rank_norm_y: bool = True
    ) -> None:
        assert regression_type in ["logit", "ols"]
        self.regression_type = regression_type
        self.logit_threshold = logit_threshold
        self.rank_norm_x = rank_norm_x
        self.ols_rank_norm_y = ols_rank_norm_y
        self.r2_name = "McFadden $R^2$" if regression_type == "logit" else "$R^2$"
        self.mod = None
        self.is_mean = None
        self.is_mean_by_stock = None

    def transform_y(self, y: pd.Series) -> np.ndarray:
        """
        Transform the dependent variable for logit or OLS.
        """
        y = y.copy()
        if self.regression_type == "logit":
            return np.where(y > self.logit_threshold, 1, 0)
        if self.regression_type == "ols" and self.ols_rank_norm_y:
            return np.array(y.groupby("Date").transform(ut.rank_normalization))
        return np.array(y)

    def transform_x(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Transform regressors by rank normalization if specified.
        """
        x = x.copy()
        if self.rank_norm_x:
            x = x.groupby("Date").transform(ut.rank_normalization)
        return x

    def fit(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Fit the specified model: logistic or OLS.
        """
        X = self.transform_x(X)
        X["const"] = 1
        if self.regression_type == "logit":
            y = self.transform_y(y)
            model = sm.Logit(y, X, missing="drop").fit(disp=0)
        else:
            y = self.transform_y(y)
            model = sm.OLS(y, X, missing="drop").fit(disp=0)

        self.mod = model
        self.is_mean = np.nanmean(y)
        return {
            "model": self.mod,
            "params": self.mod.params,
            "tstats": self.mod.tvalues,
            "pvalues": self.mod.pvalues
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict with the fitted model.
        """
        X = self.transform_x(X)
        X["const"] = 1
        return np.array(self.mod.predict(X))

    def is_r2(self) -> float:
        """
        In-sample R^2 metric from Statsmodels object.
        """
        if self.regression_type == "logit":
            return self.mod.prsquared
        return self.mod.rsquared


def load_cnn_and_monthly_stock_char(data_type: str) -> pd.DataFrame:
    """
    Utility function to load CNN and monthly characteristics (copied from analysis_lib).
    """
    save_path = CACHE_DIR / f"cnn_and_monthly_stock_char_{data_type}.parquet"
    print(f"loading from {save_path}")
    df = pd.read_parquet(save_path)
    return df
