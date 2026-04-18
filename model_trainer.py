"""
model_trainer.py
Logistic Regression training, prediction, dataset balancing, and prediction rate calculation.
Supports multi-axis balancing (gender, caste, region).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

FEATURES = ["experience", "education"]
TARGET   = "selected"

# All columns that are sensitive / should never be model features
SENSITIVE_COLS = {"gender", "caste", "region"}


def train_model(df: pd.DataFrame, features: list = None, target: str = TARGET):
    """
    Train a LogisticRegression model.
    Returns (fitted_model, predictions_array).
    """
    if features is None:
        features = [f for f in FEATURES if f in df.columns]

    X = df[features].values
    y = df[target].values

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model, model.predict(X)


def balance_dataset(df: pd.DataFrame, group_cols: list = None) -> pd.DataFrame:
    """
    Balance the dataset across one or more group columns by resampling minorities
    to match the largest group count.
    group_cols defaults to all sensitive columns present in df.
    Returns a new balanced DataFrame (shuffled).
    """
    if group_cols is None:
        group_cols = [c for c in SENSITIVE_COLS if c in df.columns]

    balanced = df.copy()
    for col in group_cols:
        groups    = balanced[col].unique()
        group_dfs = {g: balanced[balanced[col] == g] for g in groups}
        max_count = max(len(gdf) for gdf in group_dfs.values())

        parts = []
        for g, gdf in group_dfs.items():
            if len(gdf) < max_count:
                parts.append(resample(gdf, replace=True, n_samples=max_count, random_state=42))
            else:
                parts.append(gdf)
        balanced = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced


def get_model_prediction_rates(
    df: pd.DataFrame, predictions: np.ndarray, group_col: str = "gender"
) -> dict:
    """
    Compute the rate at which the model predicts positive per group.
    Returns dict mapping group label -> float rate in [0, 1].
    """
    temp = df.copy()
    temp["_pred"] = predictions
    rates = {}
    for group, subset in temp.groupby(group_col):
        total = len(subset)
        if total == 0:
            continue
        rates[str(group)] = float(subset["_pred"].sum()) / float(total)
    return rates


def get_feature_importance(model, features: list = None) -> dict:
    """
    Extract feature importance (absolute coefficients) from a fitted LogisticRegression.
    Returns dict mapping feature name -> absolute coefficient value.
    """
    if features is None:
        features = FEATURES
    coefs = np.abs(model.coef_[0])
    return {feat: float(coef) for feat, coef in zip(features, coefs)}


def remove_sensitive_and_retrain(
    df: pd.DataFrame,
    sensitive_cols: list = None,
    target: str = TARGET,
):
    """
    Retrain the model WITHOUT any sensitive attributes (gender, caste, region).
    Returns (fitted_model, predictions_array).
    """
    if sensitive_cols is None:
        sensitive_cols = list(SENSITIVE_COLS)
    safe_features = [f for f in FEATURES if f in df.columns and f not in sensitive_cols]
    return train_model(df, features=safe_features, target=target)
