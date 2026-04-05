# NHS EAD Forecast Research

## Objective

Minimize MSE for days 1-5 and 6-10 on the competition dev set using 3-fold expanding window CV.

## Current Best

- MSE 1-5: 0.048891
- MSE 6-10: 0.050872
- Config: exp_xgboost_ar.yaml (XGBoost with AR features)

## Constraints

- Must respect 3-day target lag (no data leakage)
- Runtime < 1 hour per full forecast
- CPU only (sklearn models)
- All features must be available at forecast time

## Data Split

```
Development Data: Mar 2023 - Sep 2025
CV Folds (expanding window):
  Fold 1: Train [Mar 2023 - Dec 2024] -> Validate [Jan - Mar 2025]
  Fold 2: Train [Mar 2023 - Mar 2025] -> Validate [Apr - Jun 2025]
  Fold 3: Train [Mar 2023 - Jun 2025] -> Validate [Jul - Sep 2025]

Assessment Period: Oct 2025 - Mar 2026 (hidden, -9999 values)
```

## Ideas to Explore

### High Priority
- [ ] Train window: try 60, 90, 120, 150 days
- [ ] Feature selection: vary L1 regularization strength
- [ ] Rolling feature windows: 3, 7, 14 days

### Medium Priority
- [ ] Additional models: GradientBoosting
- [ ] Ensemble: average top 3 models
- [ ] Day-of-week interactions with key features

### Low Priority / Future
- [ ] XGBoost (requires additional dependency)
- [ ] Deep learning (requires GPU)

## Off-Limits

- No external data sources (competition rule)
- No future feature values in predictions
- Don't modify src/*.py core logic unless necessary

## Completed Experiments

(Auto-populated by autoresearch.py)
