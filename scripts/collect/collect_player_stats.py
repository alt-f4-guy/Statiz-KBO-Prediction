"""라인업과 로스터 선수의 시즌·일별 원천 스냅샷을 수집한다."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from rich.console import Console

from pipeline_config import RAW_DATA_DIR, load_api_credentials
from player_stats_collection import (
    collect_player_snapshots,
    load_player_population,
)
from statiz_api import StatizAPI


console = Console()
SEOUL = ZoneInfo("Asia/Seoul")


def main() -> None:
    """종료 시즌은 재사용하고 현재 시즌은 실행할 때마다 새로 수집한다."""

    credentials = load_api_credentials()
    api = StatizAPI(credentials.api_key, credentials.secret)
    current_year = datetime.now(SEOUL).year
    years = range(2023, current_year + 1)

    player_numbers = load_player_population(
        RAW_DATA_DIR / "lineups.csv",
        RAW_DATA_DIR / "rosters.csv",
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
    )
    if failures:
        console.print(f"[yellow]재시도 대상 응답: {len(failures)}건[/yellow]")
    else:
        console.print("[green]선수 스냅샷 수집 완료[/green]")


if __name__ == "__main__":
    main()
