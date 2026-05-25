import os
import pandas as pd
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.status import Status

console = Console()

STADIUM_PF = {
    1003: 0.95, 4001: 1.02, 7002: 1.08, 2002: 1.05, 1001: 1.12,
    3001: 0.98, 6002: 1.03, 9002: 1.01, 5002: 1.06, 8001: 1.04, 11001: 1.00
}

def convert_ip(ip):
    try:
        ip_val = float(ip)
        frac = ip_val - int(ip_val)
        if abs(frac - 0.1) < 0.01: return int(ip_val) + 0.3333
        if abs(frac - 0.2) < 0.01: return int(ip_val) + 0.6666
        return ip_val
    except: return 0.0

def timestamp_to_year(ts):
    try:
        ts_f = float(ts)
        if ts_f > 1e11: ts_f /= 1000
        return datetime.fromtimestamp(ts_f).year
    except: return 2024

def calculate_pitcher_hybrid_stats(day_df, season_df):
    with console.status("[bold cyan]Pitcher Hybrid Stats 계산 중...", spinner="earth"):
        day_df = day_df.sort_values(['p_no', 'gameDate']).copy()
        day_df['IP_calc'] = day_df['IP'].apply(convert_ip)
        cols = ['HR', 'BB', 'HP', 'SO', 'ER', 'H']
        for c in cols: day_df[c] = pd.to_numeric(day_df[c], errors='coerce').fillna(0)
        season_prior_fip = season_df.set_index(['p_no', 'year'])['FIP'].to_dict()
        grouped = day_df.groupby(['p_no', 'year_req'])
        std_sums = grouped[['IP_calc'] + cols].cumsum()
        std_sums = std_sums.groupby([day_df['p_no'], day_df['year_req']]).shift(1).fillna(0)
        curr_fip = ((13 * std_sums['HR'] + 3 * (std_sums['BB'] + std_sums['HP']) - 2 * std_sums['SO']) / std_sums['IP_calc'].replace(0, np.nan)) + 3.10
        temp_df = pd.DataFrame({
            'p_no': day_df['p_no'], 'year_req': day_df['year_req'], 'std_ip': std_sums['IP_calc'],
            'curr_fip': curr_fip.fillna(5.2), 's_no_key': day_df['s_no_key']
        })
        
        # Optimized vector blending
        weights = (temp_df['std_ip'] / 5.0).clip(upper=1.0)
        p_fip_list = [
            season_prior_fip.get((int(p), int(y) - 1), 5.2) if pd.notna(p) and pd.notna(y) else 5.2
            for p, y in zip(temp_df['p_no'], temp_df['year_req'])
        ]
        p_fip = pd.Series(p_fip_list, index=temp_df.index)
        temp_df['hybrid_FIP'] = (weights * temp_df['curr_fip']) + ((1 - weights) * p_fip)
        temp_df['hybrid_FIP'] = temp_df['hybrid_FIP'].fillna(5.2).clip(1.5, 12.0)
        return temp_df

def calculate_batter_hybrid_stats(day_df, season_df):
    with console.status("[bold cyan]Batter Hybrid Stats 계산 중...", spinner="earth"):
        day_df = day_df.sort_values(['p_no', 'gameDate']).copy()
        cols = ['PA', 'H', 'TB', 'BB', 'HP', 'HR']
        for c in cols: day_df[c] = pd.to_numeric(day_df[c], errors='coerce').fillna(0)
        season_prior_wrc = season_df.set_index(['p_no', 'year'])['wRC+'].to_dict()
        grouped = day_df.groupby(['p_no', 'year_req'])
        std_sums = grouped[cols].cumsum()
        std_sums = std_sums.groupby([day_df['p_no'], day_df['year_req']]).shift(1).fillna(0)
        obp_std = (std_sums['H'] + std_sums['BB'] + std_sums['HP']) / std_sums['PA'].replace(0, np.nan)
        current_wrc = obp_std.fillna(0.280) * 333
        temp_df = pd.DataFrame({
            'p_no': day_df['p_no'], 'year_req': day_df['year_req'], 'std_pa': std_sums['PA'],
            'curr_wrc': current_wrc.fillna(92), 's_no_key': day_df['s_no_key']
        })
        
        # Optimized vector blending
        weights = (temp_df['std_pa'] / 10.0).clip(upper=1.0)
        prior_list = [
            season_prior_wrc.get((int(p), int(y) - 1), 92) if pd.notna(p) and pd.notna(y) else 92
            for p, y in zip(temp_df['p_no'], temp_df['year_req'])
        ]
        prior = pd.Series(prior_list, index=temp_df.index)
        temp_df['hybrid_wRC'] = (weights * temp_df['curr_wrc']) + ((1 - weights) * prior)
        return temp_df

def calculate_team_bullpen_rolling(day_df):
    with console.status("[bold cyan]Team Bullpen Rolling Stats 계산 중...", spinner="earth"):
        df = day_df.copy()
        rp_df = df[df['position'] != 1].copy()
        rp_df['IP_calc'] = rp_df['IP'].apply(convert_ip)
        cols = ['ER', 'H', 'BB', 'SO', 'HR', 'HP']
        for c in cols: rp_df[c] = pd.to_numeric(rp_df[c], errors='coerce').fillna(0)
        team_game_rp = rp_df.groupby(['t_code', 's_no_key'])[['IP_calc'] + cols].sum().reset_index()
        team_game_rp = team_game_rp.sort_values(['t_code', 's_no_key'])
        rolling_rp = team_game_rp.groupby('t_code')[['IP_calc'] + cols].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum()).fillna(0)
        team_game_rp['rp_rolling_era'] = (rolling_rp['ER'] * 9) / rolling_rp['IP_calc'].replace(0, np.nan)
        team_game_rp['rp_rolling_fip'] = ((13 * rolling_rp['HR'] + 3 * (rolling_rp['BB'] + rolling_rp['HP']) - 2 * rolling_rp['SO']) / rolling_rp['IP_calc'].replace(0, np.nan)) + 3.10
        team_game_rp['rp_rolling_k_bb'] = rolling_rp['SO'] / rolling_rp['BB'].replace(0, np.nan)
        team_game_rp['rp_rolling_era'] = team_game_rp['rp_rolling_era'].fillna(4.8).clip(2.0, 10.0)
        team_game_rp['rp_rolling_fip'] = team_game_rp['rp_rolling_fip'].fillna(4.8).clip(2.0, 10.0)
        team_game_rp['rp_rolling_k_bb'] = team_game_rp['rp_rolling_k_bb'].fillna(1.5).clip(0.5, 5.0)
        return team_game_rp[['t_code', 's_no_key', 'rp_rolling_era', 'rp_rolling_fip', 'rp_rolling_k_bb']]

def merge_features_v7(games_df, lineups_df, rosters_df, pitcher_hybrid, batter_hybrid, bullpen_rolling, is_prediction=False):
    with console.status("[bold green]Final Feature Matrix 병합 중...", spinner="monkey"):
        final_rows = []
        W_STARTER, W_BENCH = 1.0, 0.0
        p_lookup = pitcher_hybrid.set_index(['p_no', 's_no_key'])['hybrid_FIP'].to_dict()
        b_lookup = batter_hybrid.set_index(['p_no', 's_no_key'])['hybrid_wRC'].to_dict()
        rp_mom_dict = {}
        for _, r in bullpen_rolling.iterrows():
            t_key, s_key = int(float(r['t_code'])), int(float(r['s_no_key']))
            rp_mom_dict[(t_key, s_key)] = {'era': r['rp_rolling_era'], 'fip': r['rp_rolling_fip'], 'kbb': r['rp_rolling_k_bb']}
        
        rosters_df['date_key'] = rosters_df['pj_date'].astype(str).str.replace('-', '')
        roster_lookup = rosters_df.groupby(['date_key', 't_code'])['p_no'].apply(list).to_dict()
        
        # lineups_df s_no별 O(1) group lookup 처리
        lineups_df_copy = lineups_df.copy()
        lineups_df_copy['t_code_int'] = lineups_df_copy['t_code'].fillna(0).astype(int)
        lineups_df_copy['position_int'] = lineups_df_copy['position'].fillna(0).astype(int)
        lineups_df_copy['p_no_int'] = lineups_df_copy['p_no'].fillna(0).astype(int)
        
        lineups_by_s_no = {}
        for s_no, group in lineups_df_copy.groupby('s_no'):
            lineups_by_s_no[int(s_no)] = list(zip(group['t_code_int'], group['position_int'], group['p_no_int']))

        games_df = games_df.sort_values('s_no').reset_index(drop=True)
        for idx, game in games_df.iterrows():
            s_no_val = int(float(game['s_no']))
            s_no_str = str(s_no_val)
            date_key = s_no_str[:8]
            
            game_lineup = lineups_by_s_no.get(s_no_val)
            if not game_lineup: continue
            
            row = {'s_no': s_no_val, 'year': timestamp_to_year(game['gameDate']), 'park_factor': STADIUM_PF.get(game.get('s_code'), 1.0)}
            for role in ['home', 'away']:
                t_code = int(float(game[f'{role}Team']))
                row[f'{role}Team'] = t_code
                if not is_prediction: row[f'{role}Score'] = game.get(f'{role}Score', 0)
                
                # SP 필터링 (position == 1)
                sp_nos = [p_no for tc, pos, p_no in game_lineup if tc == t_code and pos == 1]
                sp_no = sp_nos[0] if sp_nos else None
                row[f'{role}_SP_FIP'] = p_lookup.get((sp_no, s_no_val), 5.2)
                
                m = rp_mom_dict.get((t_code, s_no_val), {'era': 4.8, 'fip': 4.8, 'kbb': 1.5})
                row[f'{role}_RP_ERA'], row[f'{role}_RP_FIP'], row[f'{role}_RP_K_BB'] = m['era'], m['fip'], m['kbb']
                
                current_roster = roster_lookup.get((date_key, t_code), [])
                
                # Batter 필터링 (position != 1)
                starter_nos = [p_no for tc, pos, p_no in game_lineup if tc == t_code and pos != 1]
                starter_stats = [b_lookup.get((p, s_no_val), 92) for p in starter_nos]
                starter_avg_wrc = np.mean(starter_stats) if starter_stats else 92
                
                starter_set = set(starter_nos)
                bench_nos = [p for p in current_roster if p not in starter_set]
                bench_stats = [b_lookup.get((p, s_no_val), 92) for p in bench_nos]
                bench_avg_wrc = np.mean(bench_stats) if bench_stats else 92
                
                row[f'{role}_batting_power'] = (starter_avg_wrc * W_STARTER) + (bench_avg_wrc * W_BENCH)
                
            row['sp_fip_diff'] = row['home_SP_FIP'] - row['away_SP_FIP']
            row['rp_era_diff'] = row['home_RP_ERA'] - row['away_RP_ERA']
            row['rp_fip_diff'] = row['home_RP_FIP'] - row['away_RP_FIP']
            row['batting_diff'] = row['home_batting_power'] - row['away_batting_power']
            row['total_diff'] = (row['batting_diff'] * 0.5) - (row['sp_fip_diff'] * 0.3) - (row['rp_fip_diff'] * 0.2)
            final_rows.append(row)
        return pd.DataFrame(final_rows)

def main():
    console.print("[bold magenta]🚀 v9.0 통합 피처 엔진 (가중치 최적화 대응 - 최적화 적용)[/bold magenta]")
    try:
        games = pd.read_csv("data/raw/games_master.csv", low_memory=False)
        lineups = pd.read_csv("data/raw/lineups.csv", low_memory=False)
        rosters = pd.read_csv("data/raw/rosters.csv", low_memory=False)
        day = pd.read_csv("data/processed/player_day_processed.csv", low_memory=False)
        season = pd.read_csv("data/processed/player_season_processed.csv", low_memory=False)

        for df in [games, lineups, rosters, day, season]:
            for col in ['s_no', 't_code', 'p_no', 'position']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        lineups = lineups.dropna(subset=['s_no', 't_code', 'p_no'])
        
    except Exception as e:
        console.print(f"[bold red]❌ 데이터 로드 실패: {e}[/bold red]"); return
    
    p_hybrid = calculate_pitcher_hybrid_stats(day, season)
    b_hybrid = calculate_batter_hybrid_stats(day, season)
    rp_rolling = calculate_team_bullpen_rolling(day)
    final_df = merge_features_v7(games, lineups, rosters, p_hybrid, b_hybrid, rp_rolling)
    final_df.to_csv("data/final/final_training_set_v8.csv", index=False, encoding='utf-8-sig')
    console.print("[bold green]✅ 데이터셋 구축 완료: data/final/final_training_set_v8.csv[/bold green]")

if __name__ == "__main__":
    main()
