"""실시간 예측 확률 선택, API 계약, 재시도와 불변 로그."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from statiz_api import StatizAPI, StatizAPIError


API_PAYLOAD_COLUMNS = {
    "ptt_idx",
    "s_no",
    "homeTeam",
    "awayTeam",
    "predictWinTeam",
    "percent",
    "update_time",
}


def prediction_window_is_open(
    now: pd.Timestamp,
    game_datetime: pd.Timestamp,
) -> bool:
    """경기 시작 전일 때만 신규 예측과 전송을 허용한다."""

    current = pd.Timestamp(now)
    game_time = pd.Timestamp(game_datetime)
    if current.tzinfo is None or game_time.tzinfo is None:
        raise ValueError("예측 시간 판정에는 시간대 정보가 필요합니다.")
    return bool(current.tz_convert("UTC") < game_time.tz_convert("UTC"))


def feature_prior_usage_rate(row: pd.Series | None) -> float:
    """실시간 피처 행에서 리그 사전분포 출처의 비율을 계산한다."""

    if row is None:
        return float("nan")
    source_columns = [
        column for column in row.index if str(column).endswith("_source")
    ]
    if not source_columns:
        return float("nan")
    return float(row[source_columns].eq("league_prior").mean())


def build_delivery_record(
    prediction_record: dict[str, Any],
    *,
    recorded_at: str,
    api_status: str,
    error_type: str = "",
) -> dict[str, Any]:
    """최초 예측의 감사 문맥을 보존한 최종 전달 이벤트를 만든다."""

    if api_status not in {"success", "failed"}:
        raise ValueError("api_status는 success 또는 failed여야 합니다.")
    record = {
        key: value
        for key, value in prediction_record.items()
        if key not in {"record_type", "recorded_at", "api_status", "error_type"}
    }
    return {
        "recorded_at": recorded_at,
        "record_type": "delivery",
        **record,
        "api_status": api_status,
        "error_type": error_type,
    }


def build_prediction_payload(
    *,
    ptt_idx: str,
    s_no: int,
    home_team_name: str,
    away_team_name: str,
    home_win_probability: float,
    update_time: str,
) -> dict[str, Any]:
    """API percent는 선택 팀과 무관하게 항상 홈팀 확률로 고정한다."""

    if not 0 <= home_win_probability <= 1:
        raise ValueError("홈 승리 확률은 0과 1 사이여야 합니다.")
    home_selected = home_win_probability >= 0.5
    selected_probability = (
        home_win_probability if home_selected else 1 - home_win_probability
    )
    return {
        "ptt_idx": ptt_idx,
        "s_no": int(s_no),
        "homeTeam": home_team_name,
        "awayTeam": away_team_name,
        "predictWinTeam": home_team_name if home_selected else away_team_name,
        "percent": round(float(home_win_probability) * 100, 2),
        "selected_team_probability": round(
            float(selected_probability) * 100, 2
        ),
        "update_time": update_time,
    }


def select_prediction_probability(
    *,
    primary_predictor: Callable[[], float],
    fallback_predictor: Callable[[], float],
    feature_quality_ok: bool,
    fallback_reason: str | None = None,
) -> tuple[float, str, str]:
    """피처 품질 또는 주 모델 추론 실패에만 대체 모델을 사용한다."""

    if not feature_quality_ok:
        reason = fallback_reason or "feature_quality_failed"
        return float(fallback_predictor()), "fallback_recent10", reason
    try:
        return float(primary_predictor()), "primary", ""
    except Exception as exc:
        reason = f"primary_inference_error:{exc.__class__.__name__}"
        return float(fallback_predictor()), "fallback_recent10", reason


def send_prediction_with_retry(
    api: StatizAPI,
    payload: dict[str, Any],
    *,
    max_attempts: int = 3,
    retry_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """같은 예측값을 제한된 횟수만 재전송하고 성공 여부를 반환한다."""

    api_payload = {
        key: value for key, value in payload.items() if key in API_PAYLOAD_COLUMNS
    }
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            response = api.post("prediction/savePrediction", api_payload)
            if isinstance(response, dict) and response.get("result_cd") in (
                100,
                "100",
            ):
                return response
            last_error = StatizAPIError("저장 API가 성공 코드를 반환하지 않았습니다.")
        except StatizAPIError as exc:
            last_error = exc
        if attempt + 1 < max_attempts:
            sleep(retry_delay * (attempt + 1))
    raise StatizAPIError("예측 저장 재시도 횟수를 초과했습니다.") from last_error


def append_prediction_log(path: Path, record: dict[str, Any]) -> None:
    """기존 행을 변경하지 않고 예측 레코드를 추가한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([record])
    frame.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
        encoding="utf-8-sig",
    )
