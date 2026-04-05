# NHS Acute Patient Harm Forecasting: Technical Report

## Introduction

This report describes our approach to forecasting daily estimated avoidable deaths in the Bristol NHS healthcare system. The goal is to predict mortality outcomes 1-10 days ahead using near-real-time operational data, enabling proactive management interventions to reduce emergency department (ED) admission delays.

## Data Preprocessing

The dataset contains 930 days of observations (March 2023 - September 2025) with 220 candidate explanatory variables recorded at varying frequencies. Our preprocessing pipeline:

1. **Temporal alignment**: Applied midday threshold aggregation—observations recorded before noon are assigned to the same day; those after noon to the following day
2. **Daily aggregation**: Computed daily means for each metric
3. **Missing value imputation**: Used linear interpolation with forward/backward fill for edge cases
4. **Column standardization**: Cleaned column names and ensured uniqueness

The target variable exhibits a three-day reporting lag, meaning predictions for day D+1 can only use target values up to day D-3.

## Feature Engineering

Given the reporting lag constraint, we designed features that respect temporal causality:

**Target lag features**: Created lags 3-7 of estimated avoidable deaths. These capture recent mortality patterns while respecting the minimum 3-day lag requirement.

**Rolling statistics**: Computed 7-day rolling mean of the lag-3 target to smooth short-term fluctuations and capture weekly trends.

**Day-of-week indicators**: One-hot encoded weekday features to capture weekly seasonality patterns. Hospital operations differ significantly between weekdays and weekends.

Total feature set: 234 columns including original operational metrics and engineered features.

## Model Selection

We evaluated multiple model classes through time-series cross-validation:
- Ridge regression
- Elastic Net
- Gradient Boosting
- XGBoost
- Neural networks (MLP)

**XGBoost** consistently outperformed alternatives, achieving MSE improvements of 15-25% over linear models. Key advantages include:
- Automatic feature interaction detection
- Robustness to irrelevant features
- Handling of non-linear relationships

**Training window selection**: We tested 90, 180, and 365-day windows. The 365-day window performed best, likely because:
- Captures full annual seasonality
- More training data improves generalization
- Target patterns evolve slowly enough that older data remains predictive

## Hyperparameter Configuration

Final XGBoost parameters:
- Trees: 200, Max depth: 5, Learning rate: 0.05
- Subsample: 0.8, Column sample: 0.6
- L1 regularization: 0.1, L2 regularization: 2.0
- Prediction clipping: [0, 1.5 × max observed value]

These parameters balance model complexity against overfitting, with conservative regularization to ensure stable predictions on unseen data.

## Results

Development period performance (549 rolling forecasts):

| Horizon | MSE |
|---------|-----|
| Days 1-5 | 0.0375 |
| Days 6-10 | 0.0396 |
| Combined | 0.0385 |

The slight degradation for longer horizons is expected as prediction uncertainty increases. The model maintains reasonable accuracy even at 10-day horizons.

## Key Findings

1. **Target lags dominate**: Recent mortality history is the strongest predictor, consistent with autocorrelation in the target series
2. **Weekly patterns matter**: Day-of-week features capture systematic variation in hospital operations
3. **Longer training helps**: Annual training windows outperform shorter alternatives
4. **XGBoost excels**: Tree-based methods handle the high-dimensional feature space effectively

## Limitations

- Model performance relies on target lag features; if reporting delays change, predictions may degrade
- Linear extrapolation of seasonal patterns assumes future years resemble past years
- Operational changes not captured in features (e.g., policy interventions) may cause forecast errors

## Implementation Notes

The algorithm runs in under 5 minutes on a standard desktop, well within the 1-hour computational limit. The rolling forecast approach allows natural recalibration as new data becomes available during the assessment period.
