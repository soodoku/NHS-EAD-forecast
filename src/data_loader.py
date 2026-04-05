"""Data loading utilities for NHS EAD forecasting."""

import zipfile
from pathlib import Path

import pandas as pd


def load_data(zip_path: str | Path) -> pd.DataFrame:
    """Load the forecasting challenge dataset from a zip file.

    Args:
        zip_path: Path to the zipped CSV file.

    Returns:
        DataFrame with columns: dt, metric_name, coverage, value, coverage_label, variable_type
    """
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f)
    df["dt"] = pd.to_datetime(df["dt"], format="mixed")
    return df


def split_target_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into target (outcome) and features.

    Args:
        df: Full dataset with variable_type column.

    Returns:
        Tuple of (target_df, features_df).
    """
    target = df[df["variable_type"] == "outcome"].copy()
    features = df[df["variable_type"] == "feature"].copy()
    return target, features
