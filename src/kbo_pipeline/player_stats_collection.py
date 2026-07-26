"""선수 모집단과 원천 API 스냅샷 수집 규칙."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from statiz_api import StatizAPI, StatizAPIError


SEOUL = ZoneInfo("Asia/Seoul")
SNAPSHOT_COLUMNS = [
    "p_no",
    "year_req",
    "fetched_at",
    "response_status",
    "json",
]


def load_player_population(
    lineup_path: Path,
    roster_path: Path,
) -> list[int]:
    """라인업과 1군 로스터의 합집합으로 선수 모집단을 만든다."""

    frames = []
    for path in (lineup_path, roster_path):
        if path.exists():
            frame = pd.read_csv(path, usecols=["p_no"])
            frames.append(pd.to_numeric(frame["p_no"], errors="coerce"))
    if not frames:
        raise FileNotFoundError("lineups.csv와 rosters.csv를 모두 찾을 수 없습니다.")

    player_numbers = pd.concat(frames, ignore_index=True).dropna().astype("int64")
    return sorted(player_numbers.unique().tolist())


def completed_player_years(snapshots: pd.DataFrame) -> set[tuple[int, int]]:
    """정상 응답이 보존된 선수-연도만 완료 상태로 간주한다."""

    required = {"p_no", "year_req", "response_status"}
    if snapshots.empty or not required.issubset(snapshots.columns):
        return set()
    success = snapshots.loc[
        snapshots["response_status"].eq("success"), ["p_no", "year_req"]
    ].copy()
    for column in ("p_no", "year_req"):
        success[column] = pd.to_numeric(success[column], errors="coerce")
    success = success.dropna().astype("int64").drop_duplicates()
    return set(success.itertuples(index=False, name=None))


def years_to_collect(
    *,
    p_no: int,
    years: Iterable[int],
    current_year: int,
    completed: set[tuple[int, int]],
) -> list[int]:
    """현재 시즌은 항상, 종료 시즌은 정상 스냅샷이 없을 때만 호출한다."""

    return [
        int(year)
        for year in years
        if int(year) == current_year or (p_no, int(year)) not in completed
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
    day_completed = completed_player_years(day_existing)
    season_completed = completed_player_years(season_existing)
    failures: list[dict[str, Any]] = []

    for p_no in player_numbers:
        day_years = years_to_collect(
            p_no=int(p_no),
            years=years,
            current_year=current_year,
            completed=day_completed,
        )
        season_years = years_to_collect(
            p_no=int(p_no),
            years=years,
            current_year=current_year,
            completed=season_completed,
        )
        fetched_at = datetime.now(SEOUL).isoformat()

        if season_years:
            try:
                season_response = api.get("prediction/playerSeason", {"p_no": p_no})
            except StatizAPIError as exc:
                season_response = {"error": exc.__class__.__name__, "msg": str(exc)}
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
            try:
                response = api.get(
                    "prediction/playerDay",
                    {"p_no": p_no, "year": year},
                )
            except StatizAPIError as exc:
                response = {"error": exc.__class__.__name__, "msg": str(exc)}
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
