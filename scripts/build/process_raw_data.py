"""원천 스냅샷을 중복 없는 v2 선수 데이터로 정형화한다."""

from __future__ import annotations

import json

import pandas as pd
from rich.console import Console

from pipeline_config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from raw_data_processing import (
    load_raw_files,
    parse_day_snapshots,
    parse_season_snapshots,
)


console = Console()


def main() -> None:
    player_dir = RAW_DATA_DIR / "players"
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    day_raw = load_raw_files(
        [
            player_dir / "playerDay_2023_2025.csv",
            player_dir / "playerDay_2023_2026.csv",
            player_dir / "player_day_snapshots.csv",
        ]
    )
    season_raw = load_raw_files(
        [
            player_dir / "playerSeason_2023_2025.csv",
            player_dir / "playerSeason_2023_2026.csv",
            player_dir / "player_season_snapshots.csv",
        ]
    )

    day, day_errors = parse_day_snapshots(day_raw)
    season, season_errors = parse_season_snapshots(season_raw)

    day_path = PROCESSED_DATA_DIR / "player_day_processed_v2.csv"
    season_path = PROCESSED_DATA_DIR / "player_season_processed_v2.csv"
    error_path = PROCESSED_DATA_DIR / "player_processing_errors.csv"
    report_path = PROCESSED_DATA_DIR / "player_processing_report.json"

    day.to_csv(day_path, index=False, encoding="utf-8-sig")
    season.to_csv(season_path, index=False, encoding="utf-8-sig")
    errors = pd.concat(
        [
            day_errors.assign(dataset="day"),
            season_errors.assign(dataset="season"),
        ],
        ignore_index=True,
    )
    errors.to_csv(error_path, index=False, encoding="utf-8-sig")

    report = {
        "day_raw_snapshots": int(len(day_raw)),
        "day_player_games": int(len(day)),
        "day_duplicate_keys": int(
            day.duplicated(["p_no", "s_no_key"]).sum()
        )
        if not day.empty
        else 0,
        "season_raw_snapshots": int(len(season_raw)),
        "season_player_years": int(len(season)),
        "season_duplicate_keys": int(
            season.duplicated(["p_no", "year"]).sum()
        )
        if not season.empty
        else 0,
        "parse_errors": int(len(errors)),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    console.print(f"[green]일별 기록 {len(day):,}행, 시즌 기록 {len(season):,}행[/green]")
    console.print(f"처리 보고서: {report_path}")


if __name__ == "__main__":
    main()
