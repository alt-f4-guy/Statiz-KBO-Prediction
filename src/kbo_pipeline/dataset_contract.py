"""최종 학습 데이터의 구조와 시점 무결성을 검증한다."""

from __future__ import annotations

import numpy as np
import pandas as pd


class DatasetContractError(ValueError):
    """학습을 중단해야 하는 데이터 계약 위반."""


REQUIRED_COLUMNS = {
    "s_no",
    "game_datetime",
    "feature_cutoff_datetime",
    "year",
    "homeTeam",
    "awayTeam",
    "homeScore",
    "awayScore",
}


def validate_final_dataset(data: pd.DataFrame) -> None:
    """치명적 품질 문제가 있으면 모델 학습 전에 예외를 발생시킨다."""

    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise DatasetContractError(f"필수 열 누락: {sorted(missing)}")
    if data.empty:
        raise DatasetContractError("학습 데이터가 비어 있습니다.")
    if data["s_no"].duplicated().any():
        raise DatasetContractError("s_no 중복이 존재합니다.")

    game_time = pd.to_datetime(data["game_datetime"], errors="coerce", utc=True)
    cutoff = pd.to_datetime(
        data["feature_cutoff_datetime"], errors="coerce", utc=True
    )
    if game_time.isna().any() or cutoff.isna().any():
        raise DatasetContractError("경기 또는 피처 기준시각 파싱에 실패했습니다.")
    if cutoff.ge(game_time).any():
        raise DatasetContractError("피처 기준시각이 경기 시작시각보다 이르지 않습니다.")

    scores = data[["homeScore", "awayScore"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if scores.isna().any().any() or scores.lt(0).any().any():
        raise DatasetContractError("목표 점수에 결측 또는 음수가 존재합니다.")

    home_features = {
        column.removeprefix("home_")
        for column in data.columns
        if column.startswith("home_")
    }
    away_features = {
        column.removeprefix("away_")
        for column in data.columns
        if column.startswith("away_")
    }
    if home_features != away_features:
        difference = sorted(home_features.symmetric_difference(away_features))
        raise DatasetContractError(f"홈·원정 피처 비대칭: {difference}")

    numeric = data.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        raise DatasetContractError("수치 피처에 무한대가 존재합니다.")


def build_feature_coverage(data: pd.DataFrame) -> pd.DataFrame:
    """경기별 핵심 피처 출처와 결측 여부를 감사 가능한 표로 만든다."""

    base_columns = [
        "s_no",
        "game_datetime",
        "feature_cutoff_datetime",
        "year",
        "homeTeam",
        "awayTeam",
    ]
    source_columns = [
        column
        for column in data.columns
        if column.endswith("_source") or column.endswith("_missing")
    ]
    coverage = data[[*base_columns, *source_columns]].copy()
    feature_columns = [
        column
        for column in data.columns
        if column not in REQUIRED_COLUMNS
        and not column.endswith("_source")
        and not column.endswith("_missing")
    ]
    coverage["feature_missing_count"] = data[feature_columns].isna().sum(axis=1)
    coverage["feature_missing_rate"] = coverage["feature_missing_count"] / max(
        len(feature_columns), 1
    )
    return coverage
