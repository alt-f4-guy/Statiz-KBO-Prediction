### 이 모듈은 2023년부터 2026년까지의 KBO 정규시즌 경기 결과(결과 포함)를 수집하여 `data/raw/games_master.csv`에 누적 업데이트합니다.
import os
import time
import hmac
import hashlib
import requests
import pandas as pd
from urllib.parse import quote
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

console = Console()

# 1. 경로 및 API 설정
ROOT_DIR = os.path.abspath(".")
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

class StatizAPI:
    def __init__(self, api_key, secret):
        self.api_key = api_key
        self.secret = secret
        self.base_url = "https://api.statiz.co.kr/baseballApi"

    def _make_signature(self, method, path, query_str, timestamp):
        payload = f"{method}|{path}|{query_str}|{timestamp}"
        return hmac.new(
            self.secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def call(self, path, params):
        timestamp = str(int(time.time()))
        safe = "-_.!~*'()"
        sorted_params = {k: str(params[k]) for k in sorted(params.keys())}
        query_str = "&".join(f"{quote(k, safe=safe)}={quote(v, safe=safe)}" for k, v in sorted_params.items())
        signature = self._make_signature("GET", path, query_str, timestamp)
        headers = {"X-API-KEY": self.api_key, "X-TIMESTAMP": timestamp, "X-SIGNATURE": signature, "Accept": "application/json"}
        try:
            response = requests.get(f"{self.base_url}/{path}", params=sorted_params, headers=headers)
            if response.status_code == 200: return response.json()
            return None
        except: return None

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("STATIZ_API_KEY")
SECRET  = os.getenv("STATIZ_SECRET")

if not API_KEY or not SECRET:
    raise ValueError("Statiz API Key와 Secret Key가 설정되지 않았습니다. .env 파일을 확인해 주세요.")

api = StatizAPI(API_KEY, SECRET)

def run_schedule_collection():
    output_file = os.path.join(RAW_DATA_DIR, "games_master.csv")
    
    # 기존 데이터 로드 (있을 경우)
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        console.print(f"📦 [bold green]기존 데이터 로드 완료:[/bold green] {len(existing_df)}건")
    else:
        existing_df = pd.DataFrame()
        console.print("🆕 [bold yellow]새로운 games_master.csv 파일을 생성합니다.[/bold yellow]")

    all_games = []
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    if not existing_df.empty:
        years = [str(current_year)]
        months = [f"{m:02d}" for m in range(1, current_month + 1)]
    else:
        years = [str(y) for y in range(2023, current_year + 1)]
        months = [f"{m:02d}" for m in range(1, 13)]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        total_steps = len(years) * len(months)
        task = progress.add_task("[cyan]Fetching schedules...", total=total_steps)
        
        for year in years:
            for month in months:
                # 신규 파일 생성 시 올해의 미래 월은 패스
                if year == str(current_year) and int(month) > current_month:
                    progress.advance(task)
                    continue
                
                progress.update(task, description=f"[cyan]Fetching {year}-{month}...")
                data = api.call("prediction/gameSchedule", {"year": year, "month": month})
                
                if data and isinstance(data, dict):
                    for date_key, games in data.items():
                        if isinstance(games, list): all_games.extend(games)
                
                time.sleep(0.1)
                progress.advance(task)

    if all_games:
        new_df = pd.DataFrame(all_games)
        new_df['leagueType'] = new_df['leagueType'].astype(str)
        new_df = new_df[new_df['leagueType'] == '10100']
        
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
            
        combined_df = combined_df.drop_duplicates(subset=['s_no'], keep='last')
        combined_df = combined_df.sort_values(['s_no']).reset_index(drop=True)
        combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 결과 요약 테이블
        summary = Table(show_header=True, header_style="bold magenta")
        summary.add_column("항목", style="dim")
        summary.add_column("값", justify="right")
        
        summary.add_row("전체 경기 수", f"{len(combined_df)}건")
        cnt_2026 = len(combined_df[combined_df['s_no'].astype(str).str.startswith('2026')])
        summary.add_row("2026년 경기 수", f"[bold cyan]{cnt_2026}건[/bold cyan]")
        
        console.print("\n✨ [bold green]업데이트 성공![/bold green]")
        console.print(summary)
    else:
        console.print("\n[bold red]❌ 수집된 데이터가 없습니다. API 연결을 확인하세요.[/bold red]")

if __name__ == "__main__":
    run_schedule_collection()
