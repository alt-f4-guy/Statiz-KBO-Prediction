### 이 모듈은 2023년부터 2026년까지의 KBO 정규시즌 경기 결과(결과 포함)를 수집하여 `data/raw/games_master.csv`에 누적 업데이트합니다.
import time
import pandas as pd
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from pipeline_config import RAW_DATA_DIR, load_api_credentials
from io_utils import atomic_to_csv
from statiz_api import StatizAPI

console = Console()

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def schedule_months_to_fetch(
    existing: pd.DataFrame,
    now: datetime,
) -> list[tuple[str, str]]:
    """현재 월과 점수가 확정되지 않은 과거 경기 월을 반환한다."""

    current = (now.year, now.month)
    if existing.empty:
        periods = pd.period_range(
            "2023-01",
            pd.Period(f"{now.year}-{now.month:02d}", freq="M"),
            freq="M",
        )
        return [
            (str(period.year), f"{period.month:02d}")
            for period in periods
        ]

    years = pd.to_numeric(existing["year"], errors="coerce")
    months = pd.to_numeric(existing["month"], errors="coerce")
    states = pd.to_numeric(existing["state"], errors="coerce")
    unfinished = (
        existing["homeScore"].isna() | existing["awayScore"].isna()
    ) & states.ne(4)
    valid = (
        unfinished
        & years.notna()
        & months.between(1, 12)
        & ((years * 12 + months) <= (now.year * 12 + now.month))
        & ((years * 12 + months) >= (2023 * 12 + 1))
    )
    unfinished_months = set(
        zip(
            years.loc[valid].astype(int),
            months.loc[valid].astype(int),
        )
    )
    targets = sorted(unfinished_months | {current})
    return [(str(year), f"{month:02d}") for year, month in targets]


def run_schedule_collection():
    credentials = load_api_credentials()
    api = StatizAPI(credentials.api_key, credentials.secret)
    output_file = RAW_DATA_DIR / "games_master.csv"
    
    # 기존 데이터 로드 (있을 경우)
    if output_file.exists():
        existing_df = pd.read_csv(output_file)
        console.print(f"📦 [bold green]기존 데이터 로드 완료:[/bold green] {len(existing_df)}건")
    else:
        existing_df = pd.DataFrame()
        console.print("🆕 [bold yellow]새로운 games_master.csv 파일을 생성합니다.[/bold yellow]")

    all_games = []
    target_months = schedule_months_to_fetch(existing_df, datetime.now())

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            "[cyan]Fetching schedules...",
            total=len(target_months),
        )

        for year, month in target_months:
            progress.update(
                task,
                description=f"[cyan]Fetching {year}-{month}...",
            )
            data = api.get(
                "prediction/gameSchedule",
                {"year": year, "month": month},
            )
            if data and isinstance(data, dict):
                for games in data.values():
                    if isinstance(games, list):
                        all_games.extend(games)

            time.sleep(0.1)
            progress.advance(task)

    if all_games:
        new_df = pd.DataFrame(all_games)
        new_df['leagueType'] = new_df['leagueType'].astype(str)
        new_df = new_df[new_df['leagueType'] == '10100']
        
        fetched_at = pd.Timestamp.now(tz="Asia/Seoul").isoformat()
        if not existing_df.empty:
            if "result_observed_at" not in existing_df.columns:
                existing_df["result_observed_at"] = pd.NA
            previous = existing_df[
                ["s_no", "homeScore", "awayScore", "result_observed_at"]
            ].rename(
                columns={
                    "homeScore": "_previous_home_score",
                    "awayScore": "_previous_away_score",
                    "result_observed_at": "_previous_result_observed_at",
                }
            )
            previous["_known_before"] = True
            new_df = new_df.merge(previous, on="s_no", how="left")
            current_scored = (
                new_df["homeScore"].notna() & new_df["awayScore"].notna()
            )
            previous_scored = (
                new_df["_previous_home_score"].notna()
                & new_df["_previous_away_score"].notna()
            )
            transitioned = (
                new_df["_known_before"].fillna(False)
                & current_scored
                & ~previous_scored
            )
            new_df["result_observed_at"] = new_df["_previous_result_observed_at"]
            new_df.loc[
                transitioned & new_df["result_observed_at"].isna(),
                "result_observed_at",
            ] = fetched_at
            new_df.drop(
                columns=[
                    "_previous_home_score",
                    "_previous_away_score",
                    "_previous_result_observed_at",
                    "_known_before",
                ],
                inplace=True,
            )
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            new_df["result_observed_at"] = pd.NA
            combined_df = new_df

        combined_df = combined_df.drop_duplicates(subset=["s_no"], keep="last")
        combined_df = combined_df.sort_values(["s_no"]).reset_index(drop=True)
        atomic_to_csv(combined_df, output_file)
        
        # 결과 요약 테이블
        summary = Table(show_header=True, header_style="bold magenta")
        summary.add_column("항목", style="dim")
        summary.add_column("값", justify="right")
        
        summary.add_row("전체 경기 수", f"{len(combined_df)}건")
        cnt_2026 = len(combined_df[pd.to_numeric(combined_df["year"], errors="coerce") == 2026])
        summary.add_row("2026년 경기 수", f"[bold cyan]{cnt_2026}건[/bold cyan]")
        
        console.print("\n✨ [bold green]업데이트 성공![/bold green]")
        console.print(summary)
    else:
        console.print("\n[bold red]❌ 수집된 데이터가 없습니다. API 연결을 확인하세요.[/bold red]")

if __name__ == "__main__":
    run_schedule_collection()
