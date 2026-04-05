#!/usr/bin/env python
"""Entry point for NHS EAD forecasting."""

import argparse
from pathlib import Path

from src.pipeline import run_forecast_pipeline


def main():
    parser = argparse.ArgumentParser(description="NHS EAD Forecasting Model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/turingAI_forecasting_challenge_dataset.csv.zip",
        help="Path to the data zip file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ridge",
        choices=["naive", "ridge", "elasticnet"],
        help="Model type to use",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="ar",
        choices=["naive", "ar", "enhanced"],
        help="Feature engineering phase",
    )
    parser.add_argument(
        "--train-window",
        type=int,
        default=90,
        help="Training window size in days",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for forecasting (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for forecasting (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent / data_path

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent / output_dir

    mse_1_5, mse_6_10 = run_forecast_pipeline(
        data_path=data_path,
        output_dir=output_dir,
        model_type=args.model,
        phase=args.phase,
        train_window=args.train_window,
        start_date=args.start_date,
        end_date=args.end_date,
        verbose=not args.quiet,
    )

    print(f"\nFinal Results:")
    print(f"  MSE (days 1-5):  {mse_1_5:.6f}")
    print(f"  MSE (days 6-10): {mse_6_10:.6f}")


if __name__ == "__main__":
    main()
