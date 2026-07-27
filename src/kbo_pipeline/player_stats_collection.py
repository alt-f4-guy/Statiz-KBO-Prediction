"""선수 모집단과 원천 API 스냅샷 수집 규칙."""

from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from statiz_api import StatizAPI


SEOUL = ZoneInfo("Asia/Seoul")
SNAPSHOT_COLUMNS = [
    "p_no",
    "year_req",
    "fetched_at",
    "response_status",
    "json",
]


def load_game_day_player_population(
    roster_path: Path,
    games_path: Path,
    target_date: date,
) -> list[int]:
    """당일 경기 팀의 당일 이하 최신 로스터 선수만 반환한다."""

    games = pd.read_csv(
        games_path,
        usecols=[
            "year",
            "month",
            "day",
            "state",
            "homeTeam",
            "awayTeam",
        ],
    )
    game_dates = pd.to_datetime(
        games[["year", "month", "day"]],
        errors="coerce",
    ).dt.date
    states = pd.to_numeric(games["state"], errors="coerce")
    today_games = games.loc[game_dates.eq(target_date) & states.ne(4)]
    if today_games.empty:
        return []

    team_codes = (
        pd.concat(
            [today_games["homeTeam"], today_games["awayTeam"]],
            ignore_index=True,
        )
        .pipe(pd.to_numeric, errors="coerce")
        .dropna()
        .astype("int64")
        .drop_duplicates()
        .sort_values()
    )

    rosters = pd.read_csv(
        roster_path,
        usecols=["pj_date", "t_code", "p_no"],
    )
    rosters["_roster_date"] = pd.to_datetime(
        rosters["pj_date"],
        errors="coerce",
    ).dt.date
    rosters["_team_code"] = pd.to_numeric(
        rosters["t_code"],
        errors="coerce",
    )
    eligible = rosters.loc[
        rosters["_team_code"].isin(team_codes)
        & rosters["_roster_date"].le(target_date)
    ].copy()
    latest_dates = eligible.groupby("_team_code")[
        "_roster_date"
    ].transform("max")
    latest = eligible.loc[eligible["_roster_date"].eq(latest_dates)]

    found_teams = set(latest["_team_code"].dropna().astype("int64"))
    missing_teams = sorted(set(team_codes) - found_teams)
    if missing_teams:
        names = ", ".join(str(team) for team in missing_teams)
        raise ValueError(f"당일 경기 팀 로스터가 없습니다: {names}")

    player_numbers = pd.to_numeric(latest["p_no"], errors="coerce")
    return sorted(player_numbers.dropna().astype("int64").unique().tolist())


def reusable_player_years(
    snapshots: pd.DataFrame,
    *,
    current_year: int,
    target_date: date,
) -> set[tuple[int, int]]:
    """종료 시즌 성공과 오늘 성공한 현재 시즌 선수-연도를 반환한다."""

    required = {
        "p_no",
        "year_req",
        "fetched_at",
        "response_status",
    }
    if snapshots.empty or not required.issubset(snapshots.columns):
        return set()

    success = snapshots.loc[
        snapshots["response_status"].eq("success"),
        ["p_no", "year_req", "fetched_at"],
    ].copy()
    success["p_no"] = pd.to_numeric(success["p_no"], errors="coerce")
    success["year_req"] = pd.to_numeric(
        success["year_req"],
        errors="coerce",
    )
    fetched_at = pd.to_datetime(
        success["fetched_at"],
        errors="coerce",
        utc=True,
    )
    fetched_dates = fetched_at.dt.tz_convert(SEOUL).dt.date
    reusable = success.loc[
        success["year_req"].ne(current_year)
        | fetched_dates.eq(target_date),
        ["p_no", "year_req"],
    ]
    reusable = reusable.dropna().astype("int64").drop_duplicates()
    return set(reusable.itertuples(index=False, name=None))


def years_to_collect(
    *,
    p_no: int,
    years: Iterable[int],
    completed: set[tuple[int, int]],
) -> list[int]:
    """재사용 가능한 성공 스냅샷이 없는 연도만 반환한다."""

    return [
        int(year)
        for year in years
        if (p_no, int(year)) not in completed
    ]


def _is_success(response: Any) -> bool:
    if not isinstance(response, dict):
        return False
    result_code = response.get("result_cd")
    return result_code in (100, "100") and "error" not in response


def _snapshot_row(
    p_no: int,
    year: int,
    response: Any,
    *,
    fetched_at: str,
) -> dict[str, Any]:
    return {
        "p_no": int(p_no),
        "year_req": int(year),
        "fetched_at": fetched_at,
        "response_status": "success" if _is_success(response) else "error",
        "json": json.dumps(response, ensure_ascii=False),
    }


def _append_snapshots(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
    frame.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
        encoding="utf-8-sig",
    )


def _filter_season_response(response: Any, year: int) -> Any:
    """시즌 전체 응답에서 요청 연도에 해당하는 원천 항목만 보존한다."""

    if not isinstance(response, dict):
        return response
    filtered: dict[str, Any] = {}
    for section_name, section in response.items():
        if isinstance(section, dict) and isinstance(section.get("list"), list):
            items = [
                item
                for item in section["list"]
                if str(item.get("year", "")) == str(year)
            ]
            filtered[section_name] = {**section, "list": items}
        elif section_name in {"result_cd", "result_msg", "update_time", "error", "msg"}:
            filtered[section_name] = section
    return filtered


def collect_player_snapshots(
    api: StatizAPI,
    player_numbers: Iterable[int],
    years: Iterable[int],
    current_year: int,
    day_snapshot_path: Path,
    season_snapshot_path: Path,
    *,
    target_date: date | None = None,
    request_interval: float = 0.3,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """선수별 필요한 연도를 호출하고 실패도 재시도 가능한 스냅샷으로 남긴다."""

    day_existing = (
        pd.read_csv(day_snapshot_path)
        if day_snapshot_path.exists()
        else pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    )
    season_existing = (
        pd.read_csv(season_snapshot_path)
        if season_snapshot_path.exists()
        else pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    )
    collection_date = target_date or datetime.now(SEOUL).date()
    day_completed = reusable_player_years(
        day_existing,
        current_year=current_year,
        target_date=collection_date,
    )
    season_completed = reusable_player_years(
        season_existing,
        current_year=current_year,
        target_date=collection_date,
    )
    failures: list[dict[str, Any]] = []

    for p_no in player_numbers:
        day_years = years_to_collect(
            p_no=int(p_no),
            years=years,
            completed=day_completed,
        )
        season_years = years_to_collect(
            p_no=int(p_no),
            years=years,
            completed=season_completed,
        )
        fetched_at = datetime.now(SEOUL).isoformat()

        if season_years:
            season_response = api.get(
                "prediction/playerSeason",
                {"p_no": p_no},
            )
            sleep(request_interval)
            season_rows = [
                _snapshot_row(
                    int(p_no),
                    year,
                    _filter_season_response(season_response, year),
                    fetched_at=fetched_at,
                )
                for year in season_years
            ]
            _append_snapshots(season_snapshot_path, season_rows)
            failures.extend(
                row for row in season_rows if row["response_status"] != "success"
            )

        for year in day_years:
            response = api.get(
                "prediction/playerDay",
                {"p_no": p_no, "year": year},
            )
            sleep(request_interval)
            row = _snapshot_row(
                int(p_no),
                year,
                response,
                fetched_at=datetime.now(SEOUL).isoformat(),
            )
            _append_snapshots(day_snapshot_path, [row])
            if row["response_status"] != "success":
                failures.append(row)

    return failures
