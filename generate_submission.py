#!/usr/bin/env python
"""Generate competition submission using best XGBoost model."""

from pathlib import Path

from src.evaluation import (
    compute_horizon_mse,
    compute_overall_mse,
    create_mse_summary_df,
    create_pred_matrix_df,
)
from src.pipeline import (
    engineer_features,
    get_feature_columns,
    prepare_data,
    run_rolling_forecast,
)

DATA_PATH = Path(__file__).parent / "data" / "turingAI_forecasting_challenge_dataset.csv.zip"
OUTPUT_DIR = Path(__file__).parent / "submission"

XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
}

TRAIN_WINDOW = 365


def main():
    print("=" * 60)
    print("NHS EAD Forecasting - Submission Generation")
    print("Model: XGBoost with 365-day training window")
    print("=" * 60)

    print("\nLoading and preprocessing data...")
    df = prepare_data(DATA_PATH)
    print(f"Data: {len(df)} days from {df['midday_day'].min()} to {df['midday_day'].max()}")

    print("\nEngineering features...")
    df = engineer_features(
        df,
        use_target_lags=True,
        use_rolling=False,
        use_dow=True,
        use_calendar=False,
        exog_lag_cols=None,
        feature_cols=None,
        use_domain_pca=False,
    )

    feature_cols = get_feature_columns(df)
    print(f"Features: {len(feature_cols)} columns")

    y_range = df["estimated_avoidable_deaths"].dropna()
    clip_range = (0.0, y_range.max() * 1.5)

    model_kwargs = {
        **XGBOOST_PARAMS,
        "clip_range": clip_range,
    }

    print(f"\nRunning rolling forecast with {TRAIN_WINDOW}-day window...")
    pred_matrix, actual_matrix, _ = run_rolling_forecast(
        df=df,
        model_type="xgboost",
        train_window=TRAIN_WINDOW,
        feature_cols=feature_cols,
        model_kwargs=model_kwargs,
        start_date=None,
        end_date=None,
        verbose=True,
        seed=42,
    )

    print("\nComputing MSE metrics...")
    mse_1_5_arr, mse_6_10_arr = compute_horizon_mse(pred_matrix, actual_matrix)
    overall_1_5, overall_6_10 = compute_overall_mse(pred_matrix, actual_matrix)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Number of forecasts: {len(pred_matrix)}")
    print(f"Overall MSE (days 1-5):  {overall_1_5:.6f}")
    print(f"Overall MSE (days 6-10): {overall_6_10:.6f}")
    print(f"Combined MSE: {(overall_1_5 + overall_6_10) / 2:.6f}")

    print("\nSaving submission files...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pred_df = create_pred_matrix_df(pred_matrix)
    mse_df = create_mse_summary_df(mse_1_5_arr, mse_6_10_arr)

    pred_df.to_csv(OUTPUT_DIR / "pred_matrix.csv", index=False)
    mse_df.to_csv(OUTPUT_DIR / "mse_summary.csv", index=False)

    print(f"  - pred_matrix.csv ({len(pred_df)} forecasts)")
    print(f"  - mse_summary.csv")
    print(f"\nSubmission files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
