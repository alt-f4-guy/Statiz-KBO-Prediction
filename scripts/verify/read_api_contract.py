"""Statiz 다섯 읽기 엔드포인트의 최소 응답 계약을 검증한다."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from pipeline_config import PROJECT_ROOT, RAW_DATA_DIR, load_api_credentials
from statiz_api import StatizAPI, StatizAPIError


SEOUL = ZoneInfo("Asia/Seoul")
VERIFICATION_DIR = PROJECT_ROOT / "verification_logs"
ENDPOINT_REQUIRED_FIELDS = {
    "prediction/gameSchedule": {"s_no", "gameDate", "homeTeam", "awayTeam"},
    "prediction/gameLineup": {"p_no", "t_code", "position", "battingOrder"},
    "prediction/playerRoster": {"p_no"},
    "prediction/playerDay": {"p_no", "gameDate", "homeScore", "awayScore"},
    "prediction/playerSeason": {"year"},
}


def _nested_records(payload: Any, required_fields: set[str]) -> list[dict]:
    records: list[dict] = []
    if isinstance(payload, dict):
        if required_fields.intersection(payload):
            records.append(payload)
        for value in payload.values():
            records.extend(_nested_records(value, required_fields))
    elif isinstance(payload, list):
        for item in payload:
            records.extend(_nested_records(item, required_fields))
    return records


def _nested_keys(payload: Any) -> set[str]:
    if isinstance(payload, dict):
        return set(payload).union(
            *(_nested_keys(value) for value in payload.values()),
            set(),
        )
    if isinstance(payload, list):
        return set().union(
            *(_nested_keys(item) for item in payload),
            set(),
        )
    return set()


def summarize_response(
    endpoint: str,
    response: Any,
    required_fields: set[str],
) -> dict[str, Any]:
    """원문과 인증 정보를 제외한 응답 계약 요약만 만든다."""

    keys = _nested_keys(response)
    missing = sorted(required_fields.difference(keys))
    result_code = response.get("result_cd") if isinstance(response, dict) else None
    return {
        "endpoint": endpoint,
        "called_at": datetime.now(SEOUL).isoformat(),
        "http_status": 200,
        "result_cd": result_code,
        "row_count": len(_nested_records(response, required_fields)),
        "required_fields": sorted(required_fields),
        "missing_fields": missing,
        "required_fields_present": not missing,
    }


def select_smoke_targets(
    games: pd.DataFrame,
    lineups: pd.DataFrame,
    rosters: pd.DataFrame,
) -> dict[str, Any]:
    """완료 경기와 해당 투수·타자·로스터를 읽기 검증 표본으로 고정한다."""

    required_game_columns = {
        "s_no",
        "year",
        "month",
        "day",
        "homeTeam",
        "homeScore",
        "awayScore",
    }
    if missing := required_game_columns.difference(games.columns):
        raise ValueError(f"경기 표본 열 누락: {sorted(missing)}")
    completed = games.loc[
        games["homeScore"].notna()
        & games["awayScore"].notna()
        & games["s_no"].isin(lineups["s_no"])
    ].copy()
    if completed.empty:
        raise ValueError("라인업이 있는 종료 경기 표본이 없습니다.")
    numeric_order = pd.DataFrame(
        {
            "_index": completed.index,
            "year": pd.to_numeric(completed["year"], errors="coerce").to_numpy(),
            "month": pd.to_numeric(
                completed["month"], errors="coerce"
            ).to_numpy(),
            "day": pd.to_numeric(completed["day"], errors="coerce").to_numpy(),
            "s_no": pd.to_numeric(completed["s_no"], errors="coerce").to_numpy(),
        }
    )
    game_index = numeric_order.sort_values(
        ["year", "month", "day", "s_no"]
    )["_index"].iloc[-1]
    game = completed.loc[game_index]
    s_no = int(game["s_no"])
    game_lineups = lineups.loc[lineups["s_no"].eq(s_no)].copy()

    position = pd.to_numeric(game_lineups["position"], errors="coerce")
    batting_order = game_lineups["battingOrder"].astype(str)
    pitchers = game_lineups.loc[position.eq(1) | batting_order.eq("P")]
    batters = game_lineups.loc[
        ~position.eq(1)
        & ~batting_order.eq("P")
        & pd.to_numeric(batting_order, errors="coerce").notna()
    ]
    if pitchers.empty or batters.empty:
        raise ValueError("종료 경기의 투수·타자 표본을 모두 찾지 못했습니다.")

    game_date = pd.Timestamp(
        year=int(game["year"]),
        month=int(game["month"]),
        day=int(game["day"]),
    ).strftime("%Y-%m-%d")
    roster_team = int(game["homeTeam"])
    roster_dates = pd.to_datetime(rosters["pj_date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    roster_teams = pd.to_numeric(rosters["t_code"], errors="coerce")
    roster_sample = rosters.loc[
        roster_dates.eq(game_date) & roster_teams.eq(roster_team)
    ]
    if roster_sample.empty:
        raise ValueError("종료 경기와 같은 팀·날짜의 로스터 표본이 없습니다.")

    return {
        "s_no": s_no,
        "year": int(game["year"]),
        "month": f"{int(game['month']):02d}",
        "pitcher_p_no": int(pitchers.iloc[0]["p_no"]),
        "batter_p_no": int(batters.iloc[0]["p_no"]),
        "roster_team": roster_team,
        "roster_date": game_date,
    }


def main() -> None:
    credentials = load_api_credentials()
    api = StatizAPI(credentials.api_key, credentials.secret)
    games = pd.read_csv(RAW_DATA_DIR / "games_master.csv")
    lineups = pd.read_csv(RAW_DATA_DIR / "lineups.csv")
    rosters = pd.read_csv(RAW_DATA_DIR / "rosters.csv")
    targets = select_smoke_targets(games, lineups, rosters)

    calls = [
        (
            "prediction/gameSchedule",
            {"year": targets["year"], "month": targets["month"]},
            ENDPOINT_REQUIRED_FIELDS["prediction/gameSchedule"],
        ),
        (
            "prediction/gameLineup",
            {"s_no": targets["s_no"]},
            ENDPOINT_REQUIRED_FIELDS["prediction/gameLineup"],
        ),
        (
            "prediction/playerRoster",
            {
                "t_code": targets["roster_team"],
                "date": targets["roster_date"],
            },
            ENDPOINT_REQUIRED_FIELDS["prediction/playerRoster"],
        ),
        (
            "prediction/playerDay",
            {
                "p_no": targets["pitcher_p_no"],
                "year": targets["year"],
            },
            ENDPOINT_REQUIRED_FIELDS["prediction/playerDay"],
        ),
        (
            "prediction/playerSeason",
            {"p_no": targets["batter_p_no"]},
            ENDPOINT_REQUIRED_FIELDS["prediction/playerSeason"],
        ),
    ]

    summaries = []
    for endpoint, params, required_fields in calls:
        try:
            response = api.get(endpoint, params)
            summary = summarize_response(endpoint, response, required_fields)
        except StatizAPIError as exc:
            summary = {
                "endpoint": endpoint,
                "called_at": datetime.now(SEOUL).isoformat(),
                "http_status": None,
                "result_cd": None,
                "row_count": 0,
                "required_fields": sorted(required_fields),
                "missing_fields": sorted(required_fields),
                "required_fields_present": False,
                "error_type": exc.__class__.__name__,
            }
        summaries.append(summary)

    VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    report_path = VERIFICATION_DIR / "read_api_contract.json"
    report_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(SEOUL).isoformat(),
                "targets": targets,
                "endpoints": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    failures = [
        item
        for item in summaries
        if not item["required_fields_present"]
        or item.get("result_cd") not in (None, 100, "100")
    ]
    if failures:
        failed_names = ", ".join(item["endpoint"] for item in failures)
        raise RuntimeError(f"읽기 API 계약 실패: {failed_names}")
    print(f"읽기 API 계약 5개 통과: {report_path}")


if __name__ == "__main__":
    main()
