"""
data_loader.py
Handles CSV loading, validation, and sample data access.
"""

import pandas as pd
from sample_data import get_sample_data

# Columns that MUST be present
REQUIRED_COLUMNS = {"gender", "experience", "education", "selected"}

# Columns that enable extended bias analysis if present
OPTIONAL_BIAS_COLUMNS = {"caste", "region"}


def load_csv(file) -> pd.DataFrame:
    """Load a CSV file-like object into a DataFrame."""
    return pd.read_csv(file)


def load_sample_data() -> pd.DataFrame:
    """Return the built-in sample hiring dataset (includes caste + region)."""
    return get_sample_data()


def validate_columns(df: pd.DataFrame) -> tuple:
    """
    Check that the DataFrame contains all required columns.
    Returns (True, "") on success or (False, error_message) on failure.
    Optional columns (caste, region) are not required but enable richer analysis.
    """
    if len(df.columns) == 0:
        return False, f"Missing required columns: {', '.join(sorted(REQUIRED_COLUMNS))}"
    cols_lower = set(df.columns.str.lower())
    missing = REQUIRED_COLUMNS - cols_lower
    if missing:
        return False, f"Missing required columns: {', '.join(sorted(missing))}"
    df.columns = df.columns.str.lower()
    return True, ""


def get_available_bias_axes(df: pd.DataFrame) -> list:
    """
    Return list of bias axes available in the DataFrame.
    Always includes 'gender'. Adds 'caste' and 'region' if those columns exist.
    """
    axes = ["gender"]
    cols = set(df.columns.str.lower())
    for col in ["caste", "region"]:
        if col in cols:
            axes.append(col)
    return axes
