"""동일 날짜를 보존하는 순차 학습·검증 분할."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _seoul_dates(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce", utc=True)
    return timestamps.dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")


def make_temporal_split_manifest(
    data: pd.DataFrame,
    *,
    development_folds: int = 3,
    calibration_fraction: float = 0.20,
) -> dict:
    """2026년을 격리하고 2025년 개발·보정 구간을 날짜 단위로 나눈다."""

    required = {"s_no", "game_datetime", "year"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"시간 분할 필수 열 누락: {sorted(missing)}")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction은 0과 1 사이여야 합니다.")
    if development_folds < 1:
        raise ValueError("development_folds는 1 이상이어야 합니다.")

    frame = data[["s_no", "game_datetime", "year"]].copy()
    frame["game_datetime"] = pd.to_datetime(
        frame["game_datetime"], errors="coerce", utc=True
    )
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["game_date"] = _seoul_dates(frame["game_datetime"])
    if frame[["game_datetime", "year", "game_date"]].isna().any().any():
        raise ValueError("시간 분할 입력에 파싱할 수 없는 날짜가 있습니다.")

    initial = frame.loc[frame["year"].lt(2025)].copy()
    year_2025 = frame.loc[frame["year"].eq(2025)].copy()
    final_test = frame.loc[frame["year"].eq(2026)].copy()
    unique_dates = np.array(sorted(year_2025["game_date"].unique()))
    if len(unique_dates) < 2:
        raise ValueError("2025년 개발·보정 분할에는 최소 2개 경기일이 필요합니다.")

    calibration_days = max(1, math.ceil(len(unique_dates) * calibration_fraction))
    calibration_dates = set(unique_dates[-calibration_days:])
    development_dates = unique_dates[:-calibration_days]
    development = year_2025.loc[
        year_2025["game_date"].isin(development_dates)
    ]
    calibration = year_2025.loc[
        year_2025["game_date"].isin(calibration_dates)
    ]

    fold_date_groups = [
        dates.tolist()
        for dates in np.array_split(development_dates, development_folds)
        if len(dates)
    ]
    folds = []
    for validation_dates in fold_date_groups:
        validation_start = min(validation_dates)
        train = frame.loc[
            frame["game_date"].lt(validation_start) & frame["year"].le(2025)
        ]
        validation = development.loc[
            development["game_date"].isin(validation_dates)
        ]
        if train.empty or validation.empty:
            continue
        if train["game_datetime"].max() >= validation["game_datetime"].min():
            raise ValueError("학습 시각이 검증 시각보다 앞서지 않습니다.")
        folds.append(
            {
                "train_s_nos": train["s_no"].astype("int64").tolist(),
                "validation_s_nos": validation["s_no"].astype("int64").tolist(),
                "train_end": train["game_datetime"].max().isoformat(),
                "validation_start": validation["game_datetime"].min().isoformat(),
            }
        )

    return {
        "timezone": "Asia/Seoul",
        "initial_train_s_nos": initial["s_no"].astype("int64").tolist(),
        "development_s_nos": development["s_no"].astype("int64").tolist(),
        "calibration_s_nos": calibration["s_no"].astype("int64").tolist(),
        "final_test_s_nos": final_test["s_no"].astype("int64").tolist(),
        "development_folds": folds,
    }


def save_split_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
