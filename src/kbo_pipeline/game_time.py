"""경기 시작 시각과 피처 가용 기준시각을 표준화한다."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEOUL_TIMEZONE = "Asia/Seoul"


def _unix_to_seoul(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    seconds = numeric.where(numeric.abs().le(1e11), numeric / 1000)
    return pd.to_datetime(seconds, unit="s", utc=True, errors="coerce").dt.tz_convert(
        SEOUL_TIMEZONE
    )


def build_result_available_datetime(games: pd.DataFrame) -> pd.Series:
    """결과가 예측 입력으로 사용 가능한 최초 시각을 반환한다."""

    game_time = _unix_to_seoul(games["gameDate"])
    observed = pd.to_datetime(
        games.get(
            "result_observed_at",
            pd.Series(pd.NaT, index=games.index),
        ),
        errors="coerce",
        utc=True,
    ).dt.tz_convert(SEOUL_TIMEZONE)
    resume = _unix_to_seoul(
        games.get(
            "gameDateResume",
            pd.Series(0, index=games.index),
        ).replace(0, np.nan)
    )
    legacy_reference = resume.fillna(game_time)
    legacy_available = (
        legacy_reference.dt.normalize()
        + pd.Timedelta(days=1)
    )
    if "homeScore" in games.columns and "awayScore" in games.columns:
        scored = games["homeScore"].notna() & games["awayScore"].notna()
    else:
        scored = pd.Series(False, index=games.index)
    return observed.fillna(legacy_available).where(scored)


def build_game_datetime_reference(games: pd.DataFrame) -> pd.DataFrame:
    """원래 경기 시작 시각을 기준으로 누수 없는 피처 마감 시각을 만든다."""

    required = {"s_no", "gameDate"}
    missing = required.difference(games.columns)
    if missing:
        raise ValueError(f"경기 시각 필수 열 누락: {sorted(missing)}")

    result = games.copy()
    result["game_datetime"] = _unix_to_seoul(result["gameDate"])
    if result["game_datetime"].isna().any():
        invalid = result.loc[result["game_datetime"].isna(), "s_no"].tolist()
        raise ValueError(f"경기 시작 시각 파싱 실패: {invalid[:10]}")

    result["game_calendar_date"] = result["game_datetime"].dt.normalize()
    epsilon = pd.Timedelta(microseconds=1)
    result["feature_cutoff_datetime"] = result["game_datetime"] - epsilon
    result["result_available_datetime"] = build_result_available_datetime(result)

    # 같은 날짜·대진의 복수 경기는 모두 첫 경기 시작 직전 상태로 고정한다.
    if {"homeTeam", "awayTeam"}.issubset(result.columns):
        home = pd.to_numeric(result["homeTeam"], errors="coerce")
        away = pd.to_numeric(result["awayTeam"], errors="coerce")
        result["_team_low"] = np.minimum(home, away)
        result["_team_high"] = np.maximum(home, away)
        group_keys = ["game_calendar_date", "_team_low", "_team_high"]
        first_start = result.groupby(group_keys, dropna=False)[
            "game_datetime"
        ].transform("min")
        group_size = result.groupby(group_keys, dropna=False)["s_no"].transform(
            "size"
        )
        result.loc[group_size.gt(1), "feature_cutoff_datetime"] = (
            first_start.loc[group_size.gt(1)] - epsilon
        )
        result.drop(columns=["_team_low", "_team_high"], inplace=True)

    columns = [
        "s_no",
        "game_datetime",
        "feature_cutoff_datetime",
        "result_available_datetime",
        "game_calendar_date",
    ]
    if "gameDateResume" in result.columns:
        result["resume_datetime"] = _unix_to_seoul(
            result["gameDateResume"].replace(0, np.nan)
        )
        columns.append("resume_datetime")
    return result[columns].sort_values(["game_datetime", "s_no"]).reset_index(
        drop=True
    )



def save_game_datetime_reference(games_path: Path, output_path: Path) -> pd.DataFrame:
    """경기 원천 파일에서 시각 참조표를 생성해 저장한다."""

    games = pd.read_csv(games_path)
    reference = build_game_datetime_reference(games)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(output_path, index=False, encoding="utf-8-sig")
    return reference
