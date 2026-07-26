"""서로 다른 모델 계열을 동일한 이진 확률 계약으로 제공한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from score_models import conditional_skellam_home_probability


@dataclass
class CalibratedScoreProbabilityModel:
    """득점 모델을 분류 모델과 같은 predict_proba() 계약으로 감싼다."""

    score_model: Any
    calibrator: Any
    feature_columns: list[str]

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        frame = data[self.feature_columns]
        home_mean, away_mean = self.score_model.predict(frame)
        raw = conditional_skellam_home_probability(home_mean, away_mean)
        calibrated = self.calibrator.predict(raw)
        return np.column_stack([1 - calibrated, calibrated])
