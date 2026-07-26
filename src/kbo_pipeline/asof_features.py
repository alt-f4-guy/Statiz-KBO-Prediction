"""선수 기록을 경기 식별자가 아닌 실제 기준시각으로 결합한다."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def merge_player_asof(
    requests: pd.DataFrame,
    events: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """선수별 기준시각 직전의 최신 누적 피처를 벡터화 결합한다."""

    request_required = {"p_no", "feature_cutoff_datetime"}
    event_required = {"p_no", "event_datetime", *feature_columns}
    missing_request = request_required.difference(requests.columns)
    missing_event = event_required.difference(events.columns)
    if missing_request or missing_event:
        raise ValueError(
            f"asof 결합 열 누락: 요청={sorted(missing_request)}, "
            f"이벤트={sorted(missing_event)}"
        )

    left = requests.copy()
    right = events[["p_no", "event_datetime", *feature_columns]].copy()
    left["p_no"] = pd.to_numeric(left["p_no"], errors="coerce")
    right["p_no"] = pd.to_numeric(right["p_no"], errors="coerce")
    left["feature_cutoff_datetime"] = pd.to_datetime(
        left["feature_cutoff_datetime"], errors="coerce", utc=True
    )
    right["event_datetime"] = pd.to_datetime(
        right["event_datetime"], errors="coerce", utc=True
    )
    left["_input_order"] = np.arange(len(left))
    left_valid = left.dropna(subset=["p_no", "feature_cutoff_datetime"]).sort_values(
        ["feature_cutoff_datetime", "p_no"]
    )
    right_valid = right.dropna(subset=["p_no", "event_datetime"]).sort_values(
        ["event_datetime", "p_no"]
    )
    merged = pd.merge_asof(
        left_valid,
        right_valid,
        left_on="feature_cutoff_datetime",
        right_on="event_datetime",
        by="p_no",
        direction="backward",
        allow_exact_matches=False,
    )
    missing_left = left.loc[
        ~left.index.isin(left_valid.index)
    ].assign(**{column: np.nan for column in feature_columns})
    merged = pd.concat([merged, missing_left], ignore_index=True, sort=False)
    return (
        merged.sort_values("_input_order")
        .drop(columns=["_input_order", "event_datetime"], errors="ignore")
        .reset_index(drop=True)
    )


def mark_bullpen_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """사전 등판 이력과 선발 역할 비율로 불펜 후보를 판정한다."""

    required = {
        "p_no",
        "starter_p_no",
        "current_g",
        "current_gs",
        "prior_g",
        "prior_gs",
        "has_pitching_history",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"불펜 후보 판정 열 누락: {sorted(missing)}")

    result = candidates.copy()
    current_g = pd.to_numeric(result["current_g"], errors="coerce").fillna(0)
    current_gs = pd.to_numeric(result["current_gs"], errors="coerce").fillna(0)
    prior_g = pd.to_numeric(result["prior_g"], errors="coerce").fillna(0)
    prior_gs = pd.to_numeric(result["prior_gs"], errors="coerce").fillna(0)
    current_ratio = current_gs / current_g.replace(0, np.nan)
    prior_ratio = prior_gs / prior_g.replace(0, np.nan)
    result["role_ratio"] = np.where(
        current_g.ge(3),
        current_ratio,
        prior_ratio,
    )
    result["role_source"] = np.select(
        [current_g.ge(3), current_g.lt(3) & prior_g.gt(0)],
        ["current_season", "prior_season"],
        default="unknown",
    )
    known_reliever = result["role_ratio"].lt(0.5)
    result["is_bullpen_candidate"] = (
        result["has_pitching_history"].fillna(False).astype(bool)
        & result["p_no"].ne(result["starter_p_no"])
        & result["role_source"].ne("unknown")
        & known_reliever
    )
    return result
