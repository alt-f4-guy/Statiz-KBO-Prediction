"""v9 주 모델과 최근 10경기 대체 모델을 사용하는 실시간 예측."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timedelta
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
from prediction_progress import (
    GameProgress,
    PredictionProgressDisplay,
    create_game_progress,
)
from realtime_prediction import (
    append_prediction_log,
    build_delivery_record,
    build_offline_prediction_record,
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


def _game_progress(
    game: dict[str, Any],
    game_time: pd.Timestamp,
) -> GameProgress:
    """경기의 대진과 서울 시작 시각으로 초기 표시 상태를 만든다."""

    home_name = _team_name(game["homeTeam"], game.get("homeTeamName"))
    away_name = _team_name(game["awayTeam"], game.get("awayTeamName"))
    start_time = game_time.tz_convert(SEOUL).strftime("%H:%M")
    return create_game_progress(
        int(game["s_no"]),
        f"{away_name} @ {home_name}",
        start_time,
    )


def _local_started_games(
    games: pd.DataFrame,
    now_utc: pd.Timestamp,
) -> list[dict[str, Any]]:
    """로컬 원천 데이터에서 이미 시작한 오늘 경기만 반환한다."""

    current = pd.Timestamp(now_utc)
    if current.tzinfo is None:
        raise ValueError("로컬 일정 선택에는 시간대 정보가 필요합니다.")

    numeric = pd.to_numeric(games["gameDate"], errors="coerce")
    seconds = numeric.where(numeric.abs().le(1e11), numeric / 1000)
    starts = pd.to_datetime(
        seconds,
        unit="s",
        errors="coerce",
        utc=True,
    )
    current_utc = current.tz_convert("UTC")
    local_dates = starts.dt.tz_convert(SEOUL).dt.date
    mask = (
        starts.notna()
        & local_dates.eq(current.tz_convert(SEOUL).date())
        & starts.le(current_utc)
    )
    return games.loc[mask].to_dict(orient="records")


def _load_today_games(
    api: StatizAPI,
    games: pd.DataFrame,
    *,
    now_utc: pd.Timestamp,
) -> tuple[list[dict[str, Any]], bool]:
    """오늘 API 일정을 읽고 실패하면 시작한 로컬 경기로 대체한다."""

    current = pd.Timestamp(now_utc)
    local_now = current.tz_convert(SEOUL)
    try:
        schedule = api.get(
            "prediction/gameSchedule",
            {
                "year": local_now.strftime("%Y"),
                "month": local_now.strftime("%m"),
            },
        )
    except StatizAPIError:
        local_games = _local_started_games(games, current)
        if local_games:
            return local_games, True
        raise

    if not isinstance(schedule, dict):
        return [], False
    return list(schedule.get(local_now.strftime("%Y%m%d"), [])), False


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
    """성공적으로 전달됐거나 오프라인 예측한 경기 ID를 복원한다."""

    if history.empty or "record_type" not in history.columns:
        return set()
    rows = history.loc[
        history["record_type"].eq("offline_prediction")
        | (
            history["record_type"].eq("delivery")
            & history.get(
                "api_status", pd.Series(index=history.index, dtype="object")
            ).eq("success")
        ),
        "s_no",
    ]
    return set(pd.to_numeric(rows, errors="coerce").dropna().astype(int))


def _lineup_wait_required(
    *,
    submit_before_start: bool,
    complete: bool,
    now_utc: pd.Timestamp,
    deadline: pd.Timestamp,
) -> bool:
    """경기 전 라인업 마감 전일 때만 다음 폴링을 기다린다."""

    return submit_before_start and not complete and now_utc < deadline


def _complete_prediction(
    api: StatizAPI,
    display: PredictionProgressDisplay,
    prediction_record: dict[str, Any],
    payload: dict[str, Any],
    *,
    game_time: pd.Timestamp,
    now_utc: pd.Timestamp,
) -> tuple[dict[str, Any], bool]:
    """현재 시각에 따라 예측을 제출하거나 오프라인 기록으로 완료한다."""

    s_no = int(prediction_record["s_no"])
    probability = float(prediction_record["home_win_probability"])
    model_type = str(prediction_record["model_type"])

    if not prediction_window_is_open(now_utc, game_time):
        record = build_offline_prediction_record(
            prediction_record,
            recorded_at=datetime.now(SEOUL).isoformat(),
        )
        append_prediction_log(PREDICTION_LOG, record)
        display.advance(
            s_no,
            step=6,
            status=f"미제출 예측 완료 · 홈 승률 {probability:.1%}",
            model=model_type,
            delivery="미제출",
        )
        return record, True

    display.advance(
        s_no,
        step=5,
        status="API 제출",
    )
    try:
        send_prediction_with_retry(api, payload)
    except StatizAPIError as exc:
        record = build_delivery_record(
            prediction_record,
            recorded_at=datetime.now(SEOUL).isoformat(),
            api_status="failed",
            error_type=exc.__class__.__name__,
        )
        append_prediction_log(PREDICTION_LOG, record)
        display.advance(
            s_no,
            step=6,
            status="제출 실패",
            delivery="실패",
            error_type=exc.__class__.__name__,
        )
        return record, False

    record = build_delivery_record(
        prediction_record,
        recorded_at=datetime.now(SEOUL).isoformat(),
        api_status="success",
    )
    append_prediction_log(PREDICTION_LOG, record)
    display.advance(
        s_no,
        step=6,
        status=f"제출 완료 · 홈 승률 {probability:.1%}",
        model=model_type,
        delivery="성공",
    )
    return record, True


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


def _run_realtime_prediction_system(
    display: PredictionProgressDisplay,
) -> None:
    with display.preparation("인증정보 확인"):
        credentials = load_api_credentials(require_ptt_idx=True)
    api = StatizAPI(
        credentials.api_key,
        credentials.secret,
        max_retries=0,
    )

    with display.preparation("모델과 메타데이터 로드"):
        model = joblib.load(MODEL_DIR / "best_model.joblib")
        metadata = json.loads(
            (MODEL_DIR / "best_model_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        features = list(model.feature_columns)

    with display.preparation("운영 데이터 로드"):
        games = pd.read_csv(RAW_DATA_DIR / "games_master.csv")
        historical_lineups = pd.read_csv(RAW_DATA_DIR / "lineups.csv")
        rosters = pd.read_csv(RAW_DATA_DIR / "rosters.csv")
        day = pd.read_csv(
            PROCESSED_DATA_DIR / "player_day_processed_v2.csv"
        )
        season = pd.read_csv(
            PROCESSED_DATA_DIR / "player_season_processed_v2.csv"
        )
        training = pd.read_csv(
            FINAL_DATA_DIR / "final_training_set_v9.csv"
        )
        non_draw_training = training.loc[
            training["homeScore"].ne(training["awayScore"])
            & training["year"].lt(2026)
        ]
        league_home_rate = float(
            non_draw_training["homeScore"]
            .gt(non_draw_training["awayScore"])
            .mean()
        )

    with display.preparation("배포 정보 확인"):
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

    log_history = _load_prediction_history()
    terminal_s_nos = _terminal_game_ids(log_history)

    while True:
        try:
            today_games, used_local_schedule = _load_today_games(
                api,
                games,
                now_utc=pd.Timestamp.now(tz="UTC"),
            )
        except StatizAPIError as exc:
            next_poll = (
                datetime.now(SEOUL) + timedelta(seconds=POLL_SECONDS)
            ).strftime("%H:%M:%S")
            display.set_waiting(
                f"일정 조회 실패 · {exc.__class__.__name__} · "
                f"다음 조회 {next_poll}"
            )
            time.sleep(POLL_SECONDS)
            continue
        if not today_games:
            next_poll = (
                datetime.now(SEOUL) + timedelta(seconds=POLL_SECONDS)
            ).strftime("%H:%M:%S")
            display.set_waiting(
                f"오늘 일정 대기 · 다음 조회 {next_poll}"
            )
            time.sleep(POLL_SECONDS)
            continue

        display.set_waiting(
            "일정 API 제한 · 로컬 일정으로 오프라인 예측"
            if used_local_schedule
            else ""
        )
        pending = [
            game
            for game in today_games
            if int(game["s_no"]) not in terminal_s_nos
        ]
        if not pending:
            display.set_waiting("오늘 경기 예측 처리 완료")
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
            display.register(_game_progress(game, game_time))
            now_utc = pd.Timestamp.now(tz="UTC")
            submit_before_start = prediction_window_is_open(
                now_utc,
                game_time,
            )
            existing = _existing_prediction(log_history, s_no)

            if existing is not None:
                prediction_record = existing.to_dict()
                probability = float(existing["home_win_probability"])
                model_type = str(existing["model_type"])
                fallback_reason = str(existing.get("fallback_reason", "") or "")
                game_datetime = str(existing["game_datetime"])
                feature_cutoff = str(existing["feature_cutoff_datetime"])
            else:
                display.advance(
                    s_no,
                    step=2,
                    status="라인업 조회",
                )
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
                if _lineup_wait_required(
                    submit_before_start=submit_before_start,
                    complete=complete,
                    now_utc=pd.Timestamp.now(tz="UTC"),
                    deadline=deadline,
                ):
                    next_poll = (
                        datetime.now(SEOUL)
                        + timedelta(seconds=POLL_SECONDS)
                    ).strftime("%H:%M:%S")
                    display.advance(
                        s_no,
                        step=2,
                        status=f"라인업 대기 · 다음 조회 {next_poll}",
                    )
                    continue

                row = None
                history = target_reference
                feature_error = ""
                if complete:
                    display.advance(
                        s_no,
                        step=3,
                        status="피처 생성",
                    )
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
                if submit_before_start:
                    append_prediction_log(PREDICTION_LOG, prediction_record)
                    log_history = pd.concat(
                        [log_history, pd.DataFrame([prediction_record])],
                        ignore_index=True,
                    )

            display.advance(
                s_no,
                step=4,
                status=fallback_reason or "모델 추론 완료",
                model=model_type,
            )
            payload = build_prediction_payload(
                ptt_idx=credentials.ptt_idx or "",
                s_no=s_no,
                home_team_name=home_name,
                away_team_name=away_name,
                home_win_probability=probability,
                update_time=datetime.now(SEOUL).strftime("%Y-%m-%d %H:%M:%S"),
            )
            completion_record, terminal = _complete_prediction(
                api,
                display,
                prediction_record,
                payload,
                game_time=game_time,
                now_utc=pd.Timestamp.now(tz="UTC"),
            )
            log_history = pd.concat(
                [log_history, pd.DataFrame([completion_record])],
                ignore_index=True,
            )
            if terminal:
                terminal_s_nos.add(s_no)
        time.sleep(POLL_SECONDS)


def run_realtime_prediction_system() -> None:
    """준비 단계와 경기별 진행 상태를 표시하며 실시간 예측을 실행한다."""

    with PredictionProgressDisplay() as display:
        _run_realtime_prediction_system(display)


if __name__ == "__main__":
    run_realtime_prediction_system()
