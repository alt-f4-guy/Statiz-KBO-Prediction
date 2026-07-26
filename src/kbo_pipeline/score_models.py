"""홈·원정 득점 기대값과 Skellam 조건부 승률 비교 모델."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from catboost import CatBoostRegressor
from scipy.stats import skellam
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


def conditional_skellam_home_probability(
    home_mean: np.ndarray,
    away_mean: np.ndarray,
) -> np.ndarray:
    """무승부를 제외한 Skellam 홈 승리 조건부 확률을 계산한다."""

    home = np.clip(np.asarray(home_mean, dtype=float), 0.01, None)
    away = np.clip(np.asarray(away_mean, dtype=float), 0.01, None)
    home_win = 1 - skellam.cdf(0, home, away)
    away_win = skellam.cdf(-1, home, away)
    denominator = home_win + away_win
    return np.divide(
        home_win,
        denominator,
        out=np.full_like(home_win, 0.5, dtype=float),
        where=denominator > 0,
    )


def _categorical_columns(feature_columns: Sequence[str]) -> list[str]:
    return [
        column
        for column in ("homeTeam", "awayTeam")
        if column in feature_columns
    ]


class CatBoostPoissonScoreModel:
    def __init__(
        self,
        feature_columns: Sequence[str],
        params: dict[str, Any] | None = None,
    ) -> None:
        self.feature_columns = list(feature_columns)
        self.categorical = _categorical_columns(feature_columns)
        defaults = {
            "iterations": 400,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 5.0,
            "loss_function": "Poisson",
            "random_seed": RANDOM_STATE,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": -1,
        }
        if params:
            defaults.update(params)
        self.params = defaults
        self.models: dict[str, CatBoostRegressor] = {}
        self.numeric_medians: pd.Series | None = None

    def _prepare(self, data: pd.DataFrame, *, fitting: bool) -> pd.DataFrame:
        frame = data[self.feature_columns].copy()
        numeric = [
            column for column in self.feature_columns if column not in self.categorical
        ]
        if fitting:
            self.numeric_medians = frame[numeric].median()
        if self.numeric_medians is None:
            raise RuntimeError("득점 모델을 먼저 적합해야 합니다.")
        frame[numeric] = frame[numeric].fillna(self.numeric_medians)
        for column in self.categorical:
            frame[column] = frame[column].fillna("missing").astype(str)
        return frame

    def fit(
        self,
        data: pd.DataFrame,
        home_score: pd.Series,
        away_score: pd.Series,
    ):
        frame = self._prepare(data, fitting=True)
        for name, target in (
            ("home", home_score),
            ("away", away_score),
        ):
            model = CatBoostRegressor(**self.params)
            model.fit(
                frame,
                pd.to_numeric(target, errors="raise"),
                cat_features=self.categorical,
            )
            self.models[name] = model
        return self

    def predict(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        frame = self._prepare(data, fitting=False)
        home = np.clip(self.models["home"].predict(frame), 0.01, None)
        away = np.clip(self.models["away"].predict(frame), 0.01, None)
        return home, away


class NegativeBinomialScoreModel:
    """고정 분산 파라미터의 음이항 GLM 비교 기준선."""

    def __init__(self, feature_columns: Sequence[str]) -> None:
        self.feature_columns = list(feature_columns)
        categorical = _categorical_columns(feature_columns)
        numeric = [
            column for column in feature_columns if column not in categorical
        ]
        self.preprocessor = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    numeric,
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(strategy="most_frequent"),
                            ),
                            (
                                "one_hot",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    sparse_output=False,
                                ),
                            ),
                        ]
                    ),
                    categorical,
                ),
            ]
        )
        self.models: dict[str, Any] = {}

    def fit(
        self,
        data: pd.DataFrame,
        home_score: pd.Series,
        away_score: pd.Series,
    ):
        design = self.preprocessor.fit_transform(data[self.feature_columns])
        design = sm.add_constant(design, has_constant="add")
        for name, target in (
            ("home", home_score),
            ("away", away_score),
        ):
            model = sm.GLM(
                pd.to_numeric(target, errors="raise").to_numpy(),
                design,
                family=sm.families.NegativeBinomial(alpha=1.0),
            )
            self.models[name] = model.fit(maxiter=200, disp=0)
        return self

    def predict(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        design = self.preprocessor.transform(data[self.feature_columns])
        design = sm.add_constant(design, has_constant="add")
        home = np.clip(self.models["home"].predict(design), 0.01, None)
        away = np.clip(self.models["away"].predict(design), 0.01, None)
        return home, away


def build_score_model(kind: str, feature_columns: Sequence[str]):
    if kind == "poisson_catboost":
        return CatBoostPoissonScoreModel(feature_columns)
    if kind == "negative_binomial":
        return NegativeBinomialScoreModel(feature_columns)
    raise ValueError(f"지원하지 않는 득점 모델: {kind}")


def score_prediction_metrics(
    actual_home: pd.Series,
    actual_away: pd.Series,
    predicted_home: np.ndarray,
    predicted_away: np.ndarray,
) -> dict[str, float]:
    actual = np.concatenate([actual_home.to_numpy(), actual_away.to_numpy()])
    predicted = np.concatenate([predicted_home, predicted_away])
    variance_ratio = float(np.var(actual, ddof=1) / np.mean(actual))
    return {
        "score_mae": float(mean_absolute_error(actual, predicted)),
        "score_rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "score_variance_mean_ratio": variance_ratio,
    }
