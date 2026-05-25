import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from rich.console import Console
from rich.status import Status

console = Console()

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def process_season_stats(raw_paths: List[str], out_path: str, day_proc_path: str):
    with console.status("[bold cyan]시즌 성적 파싱 및 복구 중...", spinner="bouncingBar"):
        season_rows = []
        for raw_path in raw_paths:
            if os.path.exists(raw_path):
                raw_df = pd.read_csv(raw_path)
                valid_df = raw_df.dropna(subset=['json'])
                p_nos = valid_df['p_no'].to_numpy()
                json_strs = valid_df['json'].to_numpy()
                for p_no, json_str in zip(p_nos, json_strs):
                    try:
                        data = json.loads(json_str)
                        if 'basic' in data and 'list' in data['basic']:
                            for s in data['basic']['list']:
                                year = s.get('year')
                                if not year: continue
                                stats = s.copy()
                                stats['p_no'] = p_no
                                if 'wRCplus' in stats: stats['wRC+'] = stats.pop('wRCplus')
                                ip = float(stats.get('IP', 0))
                                if ip > 0:
                                    h, bb, hp, so, hr = float(stats.get('H', 0)), float(stats.get('BB', 0)), float(stats.get('HP', 0)), float(stats.get('SO', 0)), float(stats.get('HR', 0))
                                    stats['FIP'] = ((13 * hr + 3 * (bb + hp) - 2 * so) / ip) + 3.10
                                season_rows.append(stats)
                    except: continue

        if os.path.exists(day_proc_path):
            day_df = pd.read_csv(day_proc_path)
            cols = ['IP', 'H', 'BB', 'HP', 'SO', 'HR', 'ER', 'PA']
            for c in cols: day_df[c] = pd.to_numeric(day_df[c], errors='coerce').fillna(0)
            agg_df = day_df.groupby(['p_no', 'year_req'])[cols].sum().reset_index()
            agg_df.rename(columns={'year_req': 'year'}, inplace=True)
            
            # Optimized O(1) whip lookup using sets
            whip_exists = {
                (int(d['p_no']), int(d['year'])) 
                for d in season_rows 
                if 'WHIP' in d and 'p_no' in d and 'year' in d
            }
            
            for r in agg_df.to_dict('records'):
                p_no, year = int(r['p_no']), int(r['year'])
                if (p_no, year) in whip_exists:
                    continue
                ip = float(r['IP'])
                if ip > 0:
                    stats = r.copy()
                    hr, bb, hp, so = float(r['HR']), float(r['BB']), float(r['HP']), float(r['SO'])
                    stats['FIP'] = ((13 * hr + 3 * (bb + hp) - 2 * so) / ip) + 3.10
                    season_rows.append(stats)

        if season_rows:
            df = pd.DataFrame(season_rows)
            df = df.sort_values(['p_no', 'year', 'FIP']).drop_duplicates(subset=['p_no', 'year'], keep='first')
            df.to_csv(out_path, index=False, encoding='utf-8-sig')
            console.print(f"  [bold green]✅ 시즌 성적 복구 완료:[/bold green] {len(df)}행")

def process_day_stats(raw_paths: List[str], out_path: str):
    with console.status("[bold cyan]일별 성적 파싱 및 정형화 중...", spinner="bouncingBar"):
        all_day_rows = []
        for raw_path in raw_paths:
            if not os.path.exists(raw_path): continue
            try:
                raw_df = pd.read_csv(raw_path)
                valid_df = raw_df.dropna(subset=['json'])
                p_nos = valid_df['p_no'].to_numpy()
                json_strs = valid_df['json'].to_numpy()
                year_req_col = valid_df['year_req'].to_numpy() if 'year_req' in valid_df.columns else [None]*len(valid_df)
                for p_no, y_req, json_str in zip(p_nos, year_req_col, json_strs):
                    try:
                        data = json.loads(json_str)
                        if not isinstance(data, dict): continue
                        for key, stats in data.items():
                            if key in ['result_cd', 'result_msg', 'update_time', 'error', 'msg']: continue
                            if isinstance(stats, dict):
                                stats['p_no'], stats['year_req'], stats['s_no_key'] = p_no, y_req, key
                                all_day_rows.append(stats)
                    except: continue
            except: continue

        if all_day_rows:
            df = pd.DataFrame(all_day_rows)
            df = df.drop_duplicates(subset=['p_no', 's_no_key'], keep='last')
            sort_cols = [c for c in ['p_no', 'year_req', 'gameDate'] if c in df.columns]
            if sort_cols: df = df.sort_values(by=sort_cols)
            df.to_csv(out_path, index=False, encoding='utf-8-sig')
            console.print(f"  [bold green]✅ 일별 성적 정형화 완료:[/bold green] {len(df)}행")

def main():
    console.print("[bold magenta]🛠️ 데이터 자가 복구 및 전처리 파이프라인 가동 (최적화 적용)[/bold magenta]")
    raw_dir = os.path.join("data", "raw", "players")
    proc_dir = os.path.join("data", "processed")
    ensure_dir(proc_dir)

    day_raws = [os.path.join(raw_dir, "playerDay_2023_2025.csv"), os.path.join(raw_dir, "playerDay_2023_2026.csv")]
    day_proc = os.path.join(proc_dir, "player_day_processed.csv")
    season_raws = [os.path.join(raw_dir, "playerSeason_2023_2025.csv"), os.path.join(raw_dir, "playerSeason_2023_2026.csv")]
    season_proc = os.path.join(proc_dir, "player_season_processed.csv")

    process_day_stats(day_raws, day_proc)
    process_season_stats(season_raws, season_proc, day_proc)

    console.print("\n[bold green]✨ 전처리 파이프라인 작업이 모두 완료되었습니다![/bold green]")

if __name__ == "__main__":
    main()
