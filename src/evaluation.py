"""Evaluation utilities for NHS EAD forecasting."""

import numpy as np
import pandas as pd


def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Mean Squared Error, ignoring NaN values.

    Args:
        actual: Array of actual values.
        predicted: Array of predicted values.

    Returns:
        MSE value.
    """
    return float(np.nanmean((actual - predicted) ** 2))


def compute_horizon_mse(
    pred_matrix: np.ndarray, actual_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MSE for days 1-5 and days 6-10 for each forecast window.

    Args:
        pred_matrix: Shape (n_forecasts, 10) predicted values.
        actual_matrix: Shape (n_forecasts, 10) actual values.

    Returns:
        Tuple of (mse_1_5, mse_6_10) arrays of shape (n_forecasts,).
    """
    n_forecasts = pred_matrix.shape[0]
    mse_1_5 = np.zeros(n_forecasts)
    mse_6_10 = np.zeros(n_forecasts)

    for i in range(n_forecasts):
        mse_1_5[i] = mse(actual_matrix[i, :5], pred_matrix[i, :5])
        mse_6_10[i] = mse(actual_matrix[i, 5:], pred_matrix[i, 5:])

    return mse_1_5, mse_6_10


def compute_overall_mse(
    pred_matrix: np.ndarray, actual_matrix: np.ndarray
) -> tuple[float, float]:
    """Compute overall MSE for days 1-5 and 6-10 across all forecasts.

    This is the competition metric.

    Args:
        pred_matrix: Shape (n_forecasts, 10) predicted values.
        actual_matrix: Shape (n_forecasts, 10) actual values.

    Returns:
        Tuple of (overall_mse_1_5, overall_mse_6_10).
    """
    mse_1_5 = mse(actual_matrix[:, :5].flatten(), pred_matrix[:, :5].flatten())
    mse_6_10 = mse(actual_matrix[:, 5:].flatten(), pred_matrix[:, 5:].flatten())
    return mse_1_5, mse_6_10


def create_pred_matrix_df(pred_matrix: np.ndarray) -> pd.DataFrame:
    """Create prediction matrix DataFrame in submission format.

    Args:
        pred_matrix: Shape (n_forecasts, 10) predicted values.

    Returns:
        DataFrame with columns: forecast_id, day_1, ..., day_10.
    """
    n_forecasts = pred_matrix.shape[0]
    df = pd.DataFrame(pred_matrix, columns=[f"day_{i+1}" for i in range(10)])
    df.insert(0, "forecast_id", range(1, n_forecasts + 1))
    return df


def create_mse_summary_df(mse_1_5: np.ndarray, mse_6_10: np.ndarray) -> pd.DataFrame:
    """Create MSE summary DataFrame in submission format.

    Args:
        mse_1_5: Array of MSE values for days 1-5.
        mse_6_10: Array of MSE values for days 6-10.

    Returns:
        DataFrame with columns: forecast_id, mse_1_5, mse_6_10.
    """
    n_forecasts = len(mse_1_5)
    df = pd.DataFrame(
        {
            "forecast_id": range(1, n_forecasts + 1),
            "mse_1_5": mse_1_5,
            "mse_6_10": mse_6_10,
        }
    )
    return df
