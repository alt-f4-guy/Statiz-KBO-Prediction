"""홈 승리 조건부 확률을 직접 예측하는 분류 모델."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
EXCLUDED_FEATURES = {
    "s_no",
    "game_datetime",
    "feature_cutoff_datetime",
    "homeScore",
    "awayScore",
    "target_home_win",
}


def prepare_classifier_frame(data: pd.DataFrame) -> pd.DataFrame:
    """무승부를 제외하고 P(홈 승리 | 비무승부)의 목표값을 만든다."""

    required = {"homeScore", "awayScore", "s_no"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"분류 목표 열 누락: {sorted(missing)}")
    frame = data.copy()
    home_score = pd.to_numeric(frame["homeScore"], errors="coerce")
    away_score = pd.to_numeric(frame["awayScore"], errors="coerce")
    valid = home_score.notna() & away_score.notna() & home_score.ne(away_score)
    frame = frame.loc[valid].copy()
    frame["target_home_win"] = (
        home_score.loc[valid].gt(away_score.loc[valid]).astype("int8")
    )
    return frame.reset_index(drop=True)


def select_model_features(data: pd.DataFrame) -> list[str]:
    """시각·목표·감사용 문자열을 제외한 경기 전 모델 피처를 선택한다."""

    features = []
    for column in data.columns:
        if column in EXCLUDED_FEATURES or column.endswith("_source"):
            continue
        if pd.api.types.is_numeric_dtype(data[column]) or column in {
            "homeTeam",
            "awayTeam",
        }:
            features.append(column)
    return features


def _categorical_columns(feature_columns: Sequence[str]) -> list[str]:
    return [
        column
        for column in ("homeTeam", "awayTeam")
        if column in feature_columns
    ]


def _logistic_pipeline(feature_columns: Sequence[str]) -> Pipeline:
    categorical = _categorical_columns(feature_columns)
    numeric = [column for column in feature_columns if column not in categorical]
    transformer = ColumnTransformer(
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
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", transformer),
            (
                "model",
                LogisticRegression(
                    max_iter=2_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


class CatBoostFrameClassifier:
    """DataFrame 열 이름과 범주형 팀 코드를 보존하는 얇은 래퍼."""

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
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "random_seed": RANDOM_STATE,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": -1,
        }
        if params:
            defaults.update(params)
        self.params = defaults
        self.model: CatBoostClassifier | None = None
        self.numeric_medians: pd.Series | None = None

    def _prepare(self, data: pd.DataFrame, *, fitting: bool) -> pd.DataFrame:
        frame = data[self.feature_columns].copy()
        numeric = [
            column for column in self.feature_columns if column not in self.categorical
        ]
        if fitting:
            self.numeric_medians = frame[numeric].median()
        if self.numeric_medians is None:
            raise RuntimeError("모델을 먼저 적합해야 합니다.")
        frame[numeric] = frame[numeric].fillna(self.numeric_medians)
        for column in self.categorical:
            frame[column] = frame[column].fillna("missing").astype(str)
        return frame

    def fit(self, data: pd.DataFrame, target: pd.Series):
        frame = self._prepare(data, fitting=True)
        target_numeric = pd.to_numeric(target, errors="raise").astype(int)
        minority_rate = target_numeric.value_counts(normalize=True).min()
        params = dict(self.params)
        if minority_rate < 0.40:
            params["auto_class_weights"] = "Balanced"
        self.model = CatBoostClassifier(**params)
        self.model.fit(frame, target_numeric, cat_features=self.categorical)
        return self

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("모델을 먼저 적합해야 합니다.")
        return self.model.predict_proba(self._prepare(data, fitting=False))


def build_classifier(
    kind: str,
    feature_columns: Sequence[str],
    params: dict[str, Any] | None = None,
):
    """고정 시드의 로지스틱 기준선 또는 CatBoost 분류기를 만든다."""

    if kind == "logistic":
        if params:
            raise ValueError("로지스틱 기준선은 별도 하이퍼파라미터를 받지 않습니다.")
        return _logistic_pipeline(feature_columns)
    if kind == "catboost":
        return CatBoostFrameClassifier(feature_columns, params)
    raise ValueError(f"지원하지 않는 분류기: {kind}")


def _unregularized_logistic() -> LogisticRegression:
    return LogisticRegression(C=np.inf, random_state=RANDOM_STATE)


class SigmoidCalibrator:
    """분리된 보정 구간의 로짓에 시그모이드 보정을 적합한다."""

    def __init__(self) -> None:
        self.model = _unregularized_logistic()

    @staticmethod
    def _logit(probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def fit(self, probability: np.ndarray, target: pd.Series):
        self.model.fit(self._logit(probability), target)
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._logit(probability))[:, 1]


@dataclass
class CalibratedBinaryModel:
    base_model: Any
    calibrator: SigmoidCalibrator
    feature_columns: list[str]

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        raw = self.base_model.predict_proba(data[self.feature_columns])[:, 1]
        calibrated = self.calibrator.predict(raw)
        return np.column_stack([1 - calibrated, calibrated])


def probability_metrics(
    target: pd.Series | np.ndarray,
    home_probability: np.ndarray,
) -> dict[str, float]:
    """확률 품질 우선순위에 맞춘 공통 이진 평가 지표."""

    y = np.asarray(target, dtype=int)
    probability = np.clip(np.asarray(home_probability, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(probability / (1 - probability)).reshape(-1, 1)
    calibration = _unregularized_logistic().fit(logit, y)
    return {
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probability)),
        "calibration_intercept": float(calibration.intercept_[0]),
        "calibration_slope": float(calibration.coef_[0, 0]),
        "roc_auc": float(roc_auc_score(y, probability)),
        "accuracy": float(accuracy_score(y, probability >= 0.5)),
    }

