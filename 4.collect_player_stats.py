import os
import time
import json
import csv
import hmac
import hashlib
import requests
from urllib.parse import quote
from typing import Any, Dict, List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

console = Console()

class StatizAPI:
    def __init__(self, api_key: str, secret: str):
        self.api_key = api_key
        self.secret = secret
        self.base_url = "https://api.statiz.co.kr/baseballApi"

    def _make_signature(self, method: str, path: str, query_str: str, timestamp: str) -> str:
        payload = f"{method}|{path}|{query_str}|{timestamp}"
        return hmac.new(
            self.secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def call(self, path: str, params: Dict[str, Any]) -> Any:
        while True:
            timestamp = str(int(time.time()))
            safe = "-_.!~*'()"
            query_str = "&".join(
                f"{quote(str(k), safe=safe)}={quote(str(params[k]), safe=safe)}"
                for k in sorted(params.keys())
            )
            signature = self._make_signature("GET", path, query_str, timestamp)

            headers = {
                "X-API-KEY": self.api_key,
                "X-TIMESTAMP": timestamp,
                "X-SIGNATURE": signature,
                "Accept": "application/json"
            }

            url = f"{self.base_url}/{path}"
            try:
                r = requests.get(url, params=params, headers=headers, timeout=30)
                if r.status_code == 200:
                    return r.json()
                elif r.status_code == 429:
                    console.print(f"  [bold red][429 Error][/bold red] API Rate limit hit. Sleeping for 60 seconds...")
                    time.sleep(60)
                    continue
                else:
                    return {"error": r.status_code, "msg": r.text}
            except Exception as e:
                console.print(f"  [bold red][Request Error][/bold red] {e}. Retrying in 10 seconds...")
                time.sleep(10)

def get_unique_pnos(lineup_csv_path: str) -> List[int]:
    pnos = set()
    with open(lineup_csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('p_no'):
                try:
                    pnos.add(int(row['p_no']))
                except ValueError:
                    pass
    return sorted(list(pnos))

def append_rows_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)

    keys = set()
    for r in rows:
        keys.update(r.keys())
    fieldnames = sorted(list(keys))

    if exists:
        with open(path, "r", encoding="utf-8-sig", newline="") as rf:
            reader = csv.reader(rf)
            existing_header = next(reader, None)
        if existing_header:
            fieldnames = existing_header

    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)

def load_progress(path: str) -> Dict:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"season": {}, "day": {}}

def save_progress(path: str, progress: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def main():
    from dotenv import load_dotenv
    load_dotenv()

    API_KEY = os.getenv("STATIZ_API_KEY")
    SECRET = os.getenv("STATIZ_SECRET")

    if not API_KEY or not SECRET:
        raise ValueError("Statiz API Key와 Secret Key가 설정되지 않았습니다. .env 파일을 확인해 주세요.")

    api = StatizAPI(API_KEY, SECRET)
    years = [2023, 2024, 2025, 2026]

    lineups_path = os.path.join("data", "raw", "lineups.csv")
    out_dir = os.path.join("data", "raw", "players")
    season_csv = os.path.join(out_dir, "playerSeason_2023_2026.csv")
    day_csv = os.path.join(out_dir, "playerDay_2023_2026.csv")
    progress_file = os.path.join(out_dir, "progress_players.json")

    pnos = get_unique_pnos(lineups_path)
    console.print(f"✅ [bold green]수집 대상 선수(p_no) 총 {len(pnos)}명 확인 완료.[/bold green]")

    progress_data = load_progress(progress_file)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Collecting player stats...", total=len(pnos))

        for pno in pnos:
            progress.update(task, description=f"[cyan]Player ID {pno}...")

            # 1. Season 데이터 수집
            s_key = str(pno)
            if not progress_data["season"].get(s_key):
                res = api.call("prediction/playerSeason", {"p_no": pno})
                row = {"p_no": pno, "json": json.dumps(res, ensure_ascii=False)}
                append_rows_csv(season_csv, [row])
                progress_data["season"][s_key] = True
                save_progress(progress_file, progress_data)
                time.sleep(0.1)

            # 2. Day 데이터 수집
            for y in years:
                d_key = f"{pno}_{y}"
                if not progress_data["day"].get(d_key):
                    res = api.call("prediction/playerDay", {"p_no": pno, "year": y})
                    rows = []
                    if isinstance(res, dict) and "s_no" in res and isinstance(res["s_no"], list):
                        for g in res["s_no"]:
                            if isinstance(g, dict):
                                r = dict(g)
                                r["p_no"] = pno
                                r["year_req"] = y
                                rows.append(r)
                    else:
                        rows.append({"p_no": pno, "year_req": y, "json": json.dumps(res, ensure_ascii=False)})

                    append_rows_csv(day_csv, rows)
                    progress_data["day"][d_key] = True
                    save_progress(progress_file, progress_data)
                    time.sleep(0.1)

            progress.advance(task)

    console.print("🎉 [bold green]선수 시즌 및 일별 성적 재수집이 완료되었습니다![/bold green]")

if __name__ == "__main__":
    main()