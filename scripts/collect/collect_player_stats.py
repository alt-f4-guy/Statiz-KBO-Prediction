"""당일 경기 팀 선수의 시즌·일별 원천 스냅샷을 증분 수집한다."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from rich.console import Console

from pipeline_config import RAW_DATA_DIR, load_api_credentials
from player_stats_collection import (
    collect_player_snapshots,
    load_game_day_player_population,
)
from statiz_api import StatizAPI


console = Console()
SEOUL = ZoneInfo("Asia/Seoul")


def main() -> None:
    """당일 경기 팀의 현재·직전 시즌에서 필요한 스냅샷만 수집한다."""

    credentials = load_api_credentials()
    api = StatizAPI(credentials.api_key, credentials.secret)
    today = datetime.now(SEOUL).date()
    current_year = today.year
    years = [current_year - 1, current_year]

    player_numbers = load_game_day_player_population(
        RAW_DATA_DIR / "rosters.csv",
        RAW_DATA_DIR / "games_master.csv",
        today,
    )
    player_dir = RAW_DATA_DIR / "players"
    day_path = player_dir / "player_day_snapshots.csv"
    season_path = player_dir / "player_season_snapshots.csv"

    console.print(f"수집 대상 선수: {len(player_numbers)}명")
    failures = collect_player_snapshots(
        api,
        player_numbers,
        years,
        current_year,
        day_path,
        season_path,
        target_date=today,
    )
    if failures:
        raise RuntimeError(f"선수 스냅샷 수집 실패: {len(failures)}건")
    console.print("[green]선수 스냅샷 수집 완료[/green]")


if __name__ == "__main__":
    main()
