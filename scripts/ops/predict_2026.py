"""v9 주 모델과 최근 10경기 대체 모델을 사용하는 실시간 예측."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

from deployment import build_deployment_context, evaluation_role_for
from fallback_recent10 import recent_ten_home_probability
from feature_matrix_v9 import _prepare_games, build_feature_matrix_v9
from pipeline_config import (
    FINAL_DATA_DIR,
    MODEL_DIR,
    OPERATIONS_DIR,
    PROJECT_ROOT,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    load_api_credentials,
)
from realtime_prediction import (
    append_prediction_log,
    build_delivery_record,
    build_prediction_payload,
    feature_prior_usage_rate,
    prediction_window_is_open,
    select_prediction_probability,
    send_prediction_with_retry,
)
from statiz_api import StatizAPI, StatizAPIError


SEOUL = ZoneInfo("Asia/Seoul")
POLL_SECONDS = 60
LINEUP_DEADLINE_MINUTES = 30
PREDICTION_LOG = OPERATIONS_DIR / "prediction_log.csv"
TEAM_NAME_MAP = {
    1001: "삼성",
    2002: "KIA",
    3001: "롯데",
    5002: "LG",
    6002: "두산",
    7002: "한화",
    9002: "SSG",
    10001: "키움",
    11001: "NC",
    12001: "KT",
    4003: "키움",
    8005: "롯데",
}


def _team_name(code: Any, supplied: Any = None) -> str:
    if isinstance(supplied, str) and supplied and not supplied.isdigit():
        return supplied
    numeric = int(float(code))
    return TEAM_NAME_MAP.get(numeric, str(numeric))


def _extract_players(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        direct = [item for item in payload if isinstance(item, dict) and "p_no" in item]
        nested = [
            player
            for item in payload
            if not (isinstance(item, dict) and "p_no" in item)
            for player in _extract_players(item)
        ]
        return direct + nested
    if isinstance(payload, dict):
        return [
            player
            for key, value in payload.items()
            if key not in {"result_cd", "result_msg", "update_time"}
            for player in _extract_players(value)
        ]
    return []


def _lineup_is_complete(
    lineup: pd.DataFrame,
    home_team: int,
    away_team: int,
) -> bool:
    team_codes = pd.to_numeric(lineup.get("t_code"), errors="coerce")
    return (
        team_codes.eq(home_team).sum() >= 9
        and team_codes.eq(away_team).sum() >= 9
    )


def _load_prediction_history() -> pd.DataFrame:
    if not PREDICTION_LOG.exists():
        return pd.DataFrame()
    return pd.read_csv(PREDICTION_LOG)


def _existing_prediction(history: pd.DataFrame, s_no: int) -> pd.Series | None:
    if history.empty or "record_type" not in history.columns:
        return None
    rows = history.loc[
        history["record_type"].eq("prediction")
        & pd.to_numeric(history["s_no"], errors="coerce").eq(s_no)
    ]
    return None if rows.empty else rows.iloc[0]


def _terminal_game_ids(history: pd.DataFrame) -> set[int]:
    """성공적으로 전달됐거나 경기 시작으로 만료된 경기 ID를 복원한다."""

    if history.empty or "record_type" not in history.columns:
        return set()
    rows = history.loc[
        history["record_type"].eq("expired")
        | (
            history["record_type"].eq("delivery")
            & history.get(
                "api_status", pd.Series(index=history.index, dtype="object")
            ).eq("success")
        ),
        "s_no",
    ]
    return set(pd.to_numeric(rows, errors="coerce").dropna().astype(int))


def _prediction_context(
    target_game: dict[str, Any],
    live_lineup: pd.DataFrame,
    games: pd.DataFrame,
    historical_lineups: pd.DataFrame,
    rosters: pd.DataFrame,
    day: pd.DataFrame,
    season: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    s_no = int(target_game["s_no"])
    target = pd.DataFrame([target_game])
    combined_games = pd.concat([games, target], ignore_index=True, sort=False)
    combined_games = combined_games.drop_duplicates("s_no", keep="last")
    combined_lineups = pd.concat(
        [
            historical_lineups.loc[
                pd.to_numeric(
                    historical_lineups["s_no"], errors="coerce"
                ).ne(s_no)
            ],
            live_lineup,
        ],
        ignore_index=True,
        sort=False,
    )
    features, _, _ = build_feature_matrix_v9(
        combined_games,
        combined_lineups,
        rosters,
        day,
        season,
        include_unscored=True,
    )
    target_features = features.loc[
        pd.to_numeric(features["s_no"], errors="coerce").eq(s_no)
    ]
    if len(target_features) != 1:
        raise ValueError(f"s_no={s_no} 실시간 피처 행을 만들지 못했습니다.")
    history = _prepare_games(combined_games, include_unscored=True)
    return target_features.iloc[0], history


def _primary_quality_ok(row: pd.Series, features: list[str]) -> bool:
    required_values = row.reindex(features)
    if required_values.isna().any():
        return False
    return not (
        bool(row.get("home_sp_missing", True))
        or bool(row.get("away_sp_missing", True))
        or float(row.get("home_bat_missing", 1.0)) > 0.5
        or float(row.get("away_bat_missing", 1.0)) > 0.5
        or float(row.get("home_bullpen_candidate_count", 0)) < 1
        or float(row.get("away_bullpen_candidate_count", 0)) < 1
    )


def run_realtime_prediction_system() -> None:
    credentials = load_api_credentials(require_ptt_idx=True)
    api = StatizAPI(credentials.api_key, credentials.secret)
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    metadata = json.loads(
        (MODEL_DIR / "best_model_metadata.json").read_text(encoding="utf-8")
    )
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    deployment_context = build_deployment_context(
        PROJECT_ROOT,
        metadata,
        git_commit,
    )
    features = list(model.feature_columns)

    games = pd.read_csv(RAW_DATA_DIR / "games_master.csv")
    historical_lineups = pd.read_csv(RAW_DATA_DIR / "lineups.csv")
    rosters = pd.read_csv(RAW_DATA_DIR / "rosters.csv")
    day = pd.read_csv(PROCESSED_DATA_DIR / "player_day_processed_v2.csv")
    season = pd.read_csv(PROCESSED_DATA_DIR / "player_season_processed_v2.csv")
    training = pd.read_csv(FINAL_DATA_DIR / "final_training_set_v9.csv")
    non_draw_training = training.loc[
        training["homeScore"].ne(training["awayScore"])
        & training["year"].lt(2026)
    ]
    league_home_rate = float(
        non_draw_training["homeScore"].gt(non_draw_training["awayScore"]).mean()
    )

    today = datetime.now(SEOUL)
    day_key = today.strftime("%Y%m%d")
    year = today.strftime("%Y")
    month = today.strftime("%m")
    log_history = _load_prediction_history()
    terminal_s_nos = _terminal_game_ids(log_history)

    while True:
        try:
            schedule = api.get(
                "prediction/gameSchedule",
                {"year": year, "month": month},
            )
        except StatizAPIError as exc:
            print(f"일정 조회 실패: {exc}")
            time.sleep(POLL_SECONDS)
            continue
        if not isinstance(schedule, dict) or day_key not in schedule:
            time.sleep(POLL_SECONDS)
            continue

        today_games = schedule[day_key]
        pending = [
            game
            for game in today_games
            if int(game["s_no"]) not in terminal_s_nos
        ]
        if not pending:
            print("오늘 경기 예측 전송 완료")
            return

        for game in pending:
            s_no = int(game["s_no"])
            home_team = int(game["homeTeam"])
            away_team = int(game["awayTeam"])
            home_name = _team_name(home_team, game.get("homeTeamName"))
            away_name = _team_name(away_team, game.get("awayTeamName"))
            target_reference = _prepare_games(
                pd.concat(
                    [games, pd.DataFrame([game])],
                    ignore_index=True,
                    sort=False,
                ).drop_duplicates("s_no", keep="last"),
                include_unscored=True,
            )
            target_time = target_reference.loc[
                target_reference["s_no"].eq(s_no)
            ].iloc[0]
            game_time = pd.Timestamp(target_time["game_datetime"])
            cutoff = pd.Timestamp(target_time["feature_cutoff_datetime"])
            now_utc = pd.Timestamp.now(tz="UTC")
            if not prediction_window_is_open(now_utc, game_time):
                expired_record = {
                    "recorded_at": datetime.now(SEOUL).isoformat(),
                    "record_type": "expired",
                    "s_no": s_no,
                    "game_datetime": game_time.isoformat(),
                    "feature_cutoff_datetime": cutoff.isoformat(),
                    "api_status": "expired",
                    **deployment_context,
                }
                append_prediction_log(PREDICTION_LOG, expired_record)
                log_history = pd.concat(
                    [log_history, pd.DataFrame([expired_record])],
                    ignore_index=True,
                )
                terminal_s_nos.add(s_no)
                continue

            existing = _existing_prediction(log_history, s_no)

            if existing is not None:
                prediction_record = existing.to_dict()
                probability = float(existing["home_win_probability"])
                model_type = str(existing["model_type"])
                fallback_reason = str(existing.get("fallback_reason", "") or "")
                game_datetime = str(existing["game_datetime"])
                feature_cutoff = str(existing["feature_cutoff_datetime"])
            else:
                try:
                    lineup_payload = api.get(
                        "prediction/gameLineup", {"s_no": s_no}
                    )
                    players = _extract_players(lineup_payload)
                except StatizAPIError:
                    players = []
                live_lineup = pd.DataFrame(players)
                if not live_lineup.empty:
                    live_lineup["s_no"] = s_no

                deadline = game_time - pd.Timedelta(
                    minutes=LINEUP_DEADLINE_MINUTES
                )
                complete = (
                    not live_lineup.empty
                    and _lineup_is_complete(
                        live_lineup, home_team, away_team
                    )
                )
                if not complete and pd.Timestamp.now(tz="UTC") < deadline:
                    continue

                row = None
                history = target_reference
                feature_error = ""
                if complete:
                    try:
                        row, history = _prediction_context(
                            game,
                            live_lineup,
                            games,
                            historical_lineups,
                            rosters,
                            day,
                            season,
                        )
                    except Exception as exc:
                        feature_error = (
                            f"feature_generation_error:{exc.__class__.__name__}"
                        )

                fallback = lambda: recent_ten_home_probability(
                    history,
                    home_team=home_team,
                    away_team=away_team,
                    feature_cutoff_datetime=cutoff,
                    league_home_win_rate=league_home_rate,
                )
                quality_ok = row is not None and _primary_quality_ok(row, features)
                reason = (
                    "lineup_deadline_exceeded"
                    if not complete
                    else feature_error or "feature_quality_failed"
                )
                probability, model_type, fallback_reason = (
                    select_prediction_probability(
                        primary_predictor=lambda: model.predict_proba(
                            pd.DataFrame([row])[features]
                        )[0, 1],
                        fallback_predictor=fallback,
                        feature_quality_ok=quality_ok,
                        fallback_reason=reason,
                    )
                )
                game_datetime = game_time.isoformat()
                feature_cutoff = cutoff.isoformat()
                payload = build_prediction_payload(
                    ptt_idx=credentials.ptt_idx or "",
                    s_no=s_no,
                    home_team_name=home_name,
                    away_team_name=away_name,
                    home_win_probability=probability,
                    update_time=datetime.now(SEOUL).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                )
                prediction_record = {
                    "recorded_at": datetime.now(SEOUL).isoformat(),
                    "record_type": "prediction",
                    "s_no": s_no,
                    "game_datetime": game_datetime,
                    "feature_cutoff_datetime": feature_cutoff,
                    "model_type": model_type,
                    "model_version": metadata["selected_model"],
                    "data_version": metadata["data_version"],
                    "fallback_reason": fallback_reason,
                    "home_win_probability": probability,
                    "selected_team_probability": (
                        payload["selected_team_probability"] / 100
                    ),
                    "predicted_team": payload["predictWinTeam"],
                    "api_status": "pending",
                    "error_type": "",
                    "evaluation_role": evaluation_role_for(
                        pd.Timestamp.now(tz=SEOUL),
                        deployment_context["prospective_start_date"],
                    ),
                    "lineup_complete": bool(complete),
                    "feature_prior_usage_rate": feature_prior_usage_rate(row),
                    **deployment_context,
                }
                append_prediction_log(PREDICTION_LOG, prediction_record)
                log_history = pd.concat(
                    [log_history, pd.DataFrame([prediction_record])],
                    ignore_index=True,
                )

            payload = build_prediction_payload(
                ptt_idx=credentials.ptt_idx or "",
                s_no=s_no,
                home_team_name=home_name,
                away_team_name=away_name,
                home_win_probability=probability,
                update_time=datetime.now(SEOUL).strftime("%Y-%m-%d %H:%M:%S"),
            )
            try:
                send_prediction_with_retry(api, payload)
            except StatizAPIError as exc:
                delivery_record = build_delivery_record(
                    prediction_record,
                    recorded_at=datetime.now(SEOUL).isoformat(),
                    api_status="failed",
                    error_type=exc.__class__.__name__,
                )
                append_prediction_log(PREDICTION_LOG, delivery_record)
                log_history = pd.concat(
                    [log_history, pd.DataFrame([delivery_record])],
                    ignore_index=True,
                )
                print(f"s_no={s_no} 예측 전송 실패: {exc}")
                continue

            delivery_record = build_delivery_record(
                prediction_record,
                recorded_at=datetime.now(SEOUL).isoformat(),
                api_status="success",
            )
            append_prediction_log(PREDICTION_LOG, delivery_record)
            log_history = pd.concat(
                [log_history, pd.DataFrame([delivery_record])],
                ignore_index=True,
            )
            terminal_s_nos.add(s_no)
            print(
                f"{away_name} @ {home_name}: 홈 승률 {probability:.1%} "
                f"({model_type})"
            )
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    run_realtime_prediction_system()
