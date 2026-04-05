"""Model definitions for NHS EAD forecasting."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.preprocessing import StandardScaler


class ForecastModel(ABC):
    """Abstract base class for forecast models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        pass


class NaiveModel(ForecastModel):
    """Naive baseline: predict last observed value for all horizons."""

    def __init__(self, seed: int = 42, **kwargs):
        self.seed = seed
        self.last_value = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.last_value = y[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0] if X.ndim > 1 else 1
        return np.full(n_samples, self.last_value)


class RidgeModel(ForecastModel):
    """Ridge regression model with cross-validated alpha."""

    def __init__(
        self,
        alphas: list[float] | None = None,
        clip_range: tuple[float, float] | None = None,
        seed: int = 42,
    ):
        if alphas is None:
            alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        self.alphas = alphas
        self.clip_range = clip_range
        self.seed = seed
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X)
        self.model = RidgeCV(alphas=self.alphas, cv=5)
        self.model.fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        if self.clip_range is not None:
            preds = np.clip(preds, self.clip_range[0], self.clip_range[1])
        return preds


class ElasticNetModel(ForecastModel):
    """Elastic Net model with cross-validated alpha and l1_ratio."""

    def __init__(
        self,
        l1_ratio: list[float] | None = None,
        alphas: list[float] | None = None,
        clip_range: tuple[float, float] | None = None,
        seed: int = 42,
    ):
        if l1_ratio is None:
            l1_ratio = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
        if alphas is None:
            alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
        self.l1_ratio = l1_ratio
        self.alphas = alphas
        self.clip_range = clip_range
        self.seed = seed
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X)
        self.model = ElasticNetCV(
            l1_ratio=self.l1_ratio,
            alphas=self.alphas,
            cv=5,
            max_iter=10000,
            random_state=self.seed,
        )
        self.model.fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        if self.clip_range is not None:
            preds = np.clip(preds, self.clip_range[0], self.clip_range[1])
        return preds


class GradientBoostingModel(ForecastModel):
    """Gradient Boosting model for forecasting."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        clip_range: tuple[float, float] | None = None,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.clip_range = clip_range
        self.seed = seed
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X)
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed,
        )
        self.model.fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        if self.clip_range is not None:
            preds = np.clip(preds, self.clip_range[0], self.clip_range[1])
        return preds


class XGBoostModel(ForecastModel):
    """XGBoost model for forecasting."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        clip_range: tuple[float, float] | None = None,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.clip_range = clip_range
        self.seed = seed
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X)
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.seed,
            verbosity=0,
        )
        self.model.fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        if self.clip_range is not None:
            preds = np.clip(preds, self.clip_range[0], self.clip_range[1])
        return preds


class ProphetModel(ForecastModel):
    """Prophet time series model for forecasting.

    Note: Prophet uses a different interface (dates + target) instead of feature matrices.
    The fit() and predict() methods here are placeholders to satisfy the ABC interface,
    but the actual forecasting is done via fit_prophet() and predict_prophet().
    """

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        seasonality_mode: str = "additive",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        seed: int = 42,
        **kwargs,
    ):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.seed = seed
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError("Prophet uses fit_prophet() with dates, not feature matrices")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Prophet uses predict_prophet() with future dates, not feature matrices")

    def fit_prophet(self, dates: pd.Series, y: np.ndarray) -> None:
        """Fit Prophet model on dates and target values."""
        from prophet import Prophet

        df = pd.DataFrame({"ds": pd.to_datetime(dates), "y": y})
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
        )
        self.model.fit(df)

    def predict_prophet(self, future_dates: pd.Series) -> np.ndarray:
        """Generate predictions for future dates."""
        future = pd.DataFrame({"ds": pd.to_datetime(future_dates)})
        forecast = self.model.predict(future)
        return forecast["yhat"].values


class MLPModel(ForecastModel):
    """Multi-layer perceptron (PyTorch) for forecasting."""

    def __init__(
        self,
        hidden_sizes: list[int] | None = None,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        dropout: float = 0.1,
        clip_range: tuple[float, float] | None = None,
        seed: int = 42,
    ):
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.clip_range = clip_range
        self.seed = seed
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cpu")

    def _build_model(self, input_size: int) -> nn.Module:
        torch.manual_seed(self.seed)
        layers = []
        prev_size = input_size
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        X_scaled = self.scaler.fit_transform(X)

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)

        self.model = self._build_model(X.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.model.train()
        for _ in range(self.epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy().flatten()

        if self.clip_range is not None:
            preds = np.clip(preds, self.clip_range[0], self.clip_range[1])
        return preds


def get_model(model_type: str, **kwargs) -> ForecastModel:
    """Factory function to get a model by name.

    Args:
        model_type: One of 'naive', 'ridge', 'elasticnet', 'xgboost', 'mlp', 'prophet'.
        **kwargs: Additional arguments for the model.

    Returns:
        ForecastModel instance.
    """
    models = {
        "naive": NaiveModel,
        "ridge": RidgeModel,
        "elasticnet": ElasticNetModel,
        "gradientboosting": GradientBoostingModel,
        "xgboost": XGBoostModel,
        "mlp": MLPModel,
        "prophet": ProphetModel,
    }
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    return models[model_type](**kwargs)
