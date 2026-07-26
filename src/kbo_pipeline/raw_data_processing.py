"""원천 선수 스냅샷을 선수-경기와 선수-연도 단위로 정형화한다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


METADATA_KEYS = {
    "result_cd",
    "result_msg",
    "update_time",
    "error",
    "msg",
}


def select_latest_successful_snapshots(
    snapshots: pd.DataFrame,
    keys: Sequence[str],
) -> pd.DataFrame:
    """키별 최신 정상 응답만 반환한다."""

    required = set(keys) | {"fetched_at", "response_status"}
    missing = required.difference(snapshots.columns)
    if missing:
        raise ValueError(f"스냅샷 필수 열 누락: {sorted(missing)}")

    successful = snapshots.loc[snapshots["response_status"].eq("success")].copy()
    successful["fetched_at"] = pd.to_datetime(
        successful["fetched_at"], errors="coerce", utc=True
    )
    successful = successful.dropna(subset=[*keys, "fetched_at"])
    return (
        successful.sort_values([*keys, "fetched_at"])
        .drop_duplicates(subset=list(keys), keep="last")
        .reset_index(drop=True)
    )


def convert_baseball_innings(values: pd.Series) -> pd.Series:
    """야구식 이닝 표기 .1/.2를 각각 1/3·2/3이닝으로 변환한다."""

    numeric = pd.to_numeric(values, errors="coerce")
    whole = np.floor(numeric)
    outs = np.rint((numeric - whole) * 10)
    valid = numeric.isna() | outs.isin([0, 1, 2])
    converted = whole + outs / 3
    return converted.where(valid)


def deduplicate_player_games(games: pd.DataFrame) -> pd.DataFrame:
    """선수-경기 키별 최신 스냅샷의 값만 유지한다."""

    if games.empty:
        return games.copy()
    required = {"p_no", "s_no_key", "fetched_at"}
    missing = required.difference(games.columns)
    if missing:
        raise ValueError(f"일별 기록 필수 열 누락: {sorted(missing)}")

    result = games.copy()
    result["fetched_at"] = pd.to_datetime(
        result["fetched_at"], errors="coerce", utc=True
    )
    return (
        result.sort_values(["p_no", "s_no_key", "fetched_at"])
        .drop_duplicates(["p_no", "s_no_key"], keep="last")
        .reset_index(drop=True)
    )


def load_raw_files(paths: Iterable[Path]) -> pd.DataFrame:
    """기존 원천 파일과 새 스냅샷 파일을 읽고 출처 순서를 보존한다."""

    frames = []
    for source_order, path in enumerate(paths):
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["_source_file"] = path.name
        frame["_source_order"] = source_order
        if "fetched_at" not in frame.columns:
            # 레거시 파일에는 수집 시각이 없으므로 파일 우선순위와 행 순서를
            # 명시적인 감사용 대체 시각으로 사용한다.
            offsets = pd.to_timedelta(
                source_order * max(len(frame), 1) + np.arange(len(frame)),
                unit="s",
            )
            frame["fetched_at"] = pd.Timestamp("1970-01-01", tz="UTC") + offsets
            frame["response_status"] = "legacy"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _decode_json(value: Any) -> tuple[Any, str | None]:
    if pd.isna(value):
        return None, "json_missing"
    try:
        return json.loads(value), None
    except (TypeError, json.JSONDecodeError):
        return None, "json_decode_error"


def _usable_snapshot_rows(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    if raw["response_status"].eq("success").any():
        current = select_latest_successful_snapshots(
            raw.loc[raw["response_status"].eq("success")],
            ["p_no", "year_req"],
        )
        legacy = raw.loc[raw["response_status"].eq("legacy")]
        return pd.concat([legacy, current], ignore_index=True, sort=False)
    return raw.loc[raw["response_status"].eq("legacy")].copy()


def parse_day_snapshots(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """일별 JSON 응답을 선수-경기 행으로 펼치고 오류를 별도로 반환한다."""

    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    usable = _usable_snapshot_rows(raw)
    game_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for record in usable.itertuples(index=False):
        record_dict = record._asdict()
        payload, error = _decode_json(record_dict.get("json"))
        if error or not isinstance(payload, dict):
            errors.append(
                {
                    "source": record_dict.get("_source_file"),
                    "p_no": record_dict.get("p_no"),
                    "year_req": record_dict.get("year_req"),
                    "reason": error or "payload_not_object",
                }
            )
            continue

        entries: list[tuple[Any, Any]]
        if isinstance(payload.get("s_no"), list):
            entries = [
                (game.get("s_no"), game)
                for game in payload["s_no"]
                if isinstance(game, dict)
            ]
        else:
            entries = [
                (key, value)
                for key, value in payload.items()
                if key not in METADATA_KEYS and isinstance(value, dict)
            ]

        for key, stats in entries:
            game = dict(stats)
            game["p_no"] = record_dict.get("p_no")
            game["year_req"] = record_dict.get("year_req")
            game["s_no_key"] = game.get("s_no", key)
            game["fetched_at"] = record_dict.get("fetched_at")
            game["source_file"] = record_dict.get("_source_file")
            game_rows.append(game)

    games = pd.DataFrame(game_rows)
    if not games.empty:
        games["p_no"] = pd.to_numeric(games["p_no"], errors="coerce")
        games["s_no_key"] = pd.to_numeric(games["s_no_key"], errors="coerce")
        games = games.dropna(subset=["p_no", "s_no_key"])
        games[["p_no", "s_no_key"]] = games[["p_no", "s_no_key"]].astype("int64")
        games = deduplicate_player_games(games)
        if "IP" in games.columns:
            games["IP"] = convert_baseball_innings(games["IP"])
    return games, pd.DataFrame(errors)


def parse_season_snapshots(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """시즌 JSON의 basic/deepen 항목을 선수-연도 단위로 결합한다."""

    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    usable = _usable_snapshot_rows(raw)
    basic_rows: list[dict[str, Any]] = []
    deep_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for record in usable.itertuples(index=False):
        record_dict = record._asdict()
        payload, error = _decode_json(record_dict.get("json"))
        if error or not isinstance(payload, dict):
            errors.append(
                {
                    "source": record_dict.get("_source_file"),
                    "p_no": record_dict.get("p_no"),
                    "year_req": record_dict.get("year_req"),
                    "reason": error or "payload_not_object",
                }
            )
            continue

        for target, collection in (("basic", basic_rows), ("deepen", deep_rows)):
            section = payload.get(target, {})
            items = section.get("list", []) if isinstance(section, dict) else []
            for item in items:
                if not isinstance(item, dict):
                    continue
                row = dict(item)
                row["p_no"] = row.get("p_no", record_dict.get("p_no"))
                row["year"] = row.get("year", record_dict.get("year_req"))
                row["fetched_at"] = record_dict.get("fetched_at")
                collection.append(row)

    basic = pd.DataFrame(basic_rows)
    deepen = pd.DataFrame(deep_rows)
    for frame in (basic, deepen):
        if not frame.empty:
            frame["p_no"] = pd.to_numeric(frame["p_no"], errors="coerce")
            frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
            frame.dropna(subset=["p_no", "year"], inplace=True)
            frame[["p_no", "year"]] = frame[["p_no", "year"]].astype("int64")
            frame.sort_values(["p_no", "year", "fetched_at"], inplace=True)
            frame.drop_duplicates(["p_no", "year"], keep="last", inplace=True)

    if basic.empty:
        seasons = deepen
    elif deepen.empty:
        seasons = basic
    else:
        overlap = set(basic.columns).intersection(deepen.columns) - {
            "p_no",
            "year",
            "fetched_at",
        }
        deepen = deepen.drop(columns=sorted(overlap), errors="ignore")
        seasons = basic.merge(
            deepen,
            on=["p_no", "year"],
            how="outer",
            suffixes=("", "_deepen"),
        )
        if "fetched_at_deepen" in seasons.columns:
            seasons["fetched_at"] = seasons["fetched_at"].fillna(
                seasons.pop("fetched_at_deepen")
            )

    if not seasons.empty:
        seasons.rename(columns={"wRCplus": "wRC+"}, inplace=True)
        if "IP" in seasons.columns:
            seasons["IP"] = convert_baseball_innings(seasons["IP"])
        seasons = seasons.drop_duplicates(["p_no", "year"], keep="last")
    return seasons.reset_index(drop=True), pd.DataFrame(errors)
