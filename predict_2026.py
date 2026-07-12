import os
import time
import hmac
import hashlib
import requests
import json
import pandas as pd
import numpy as np
import warnings
import gc
from urllib.parse import quote
from datetime import datetime
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import skellam

# UI 강화를 위한 rich 라이브러리
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.status import Status

from create_feature_matrix_v7 import (
    calculate_pitcher_hybrid_stats, 
    calculate_batter_hybrid_stats, 
    calculate_team_bullpen_rolling,
    merge_features_v7
)

warnings.filterwarnings('ignore')
console = Console()

# Statiz API 설정 (.env 로드 방식 유지)
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("STATIZ_API_KEY")
SECRET  = os.getenv("STATIZ_SECRET")

if not API_KEY or not SECRET:
    raise ValueError("Statiz API Key와 Secret Key가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
PTT_IDX = "05"

class StatizAPI:
    """
    Statiz API 호출 및 서명(Signature) 인증 처리를 담당하는 클래스
    """
    def __init__(self, api_key, secret):
        self.api_key = api_key
        self.secret = secret
        self.base_url = "https://api.statiz.co.kr/baseballApi"

    def _make_signature(self, method, path, query_str, timestamp):
        payload = f"{method}|{path}|{query_str}|{timestamp}"
        return hmac.new(self.secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def call(self, path, params):
        timestamp = str(int(time.time()))
        safe = "-_.!~*'()"
        sorted_params = {k: str(params[k]) for k in sorted(params.keys())}
        query_str = "&".join(f"{quote(k, safe=safe)}={quote(v, safe=safe)}" for k, v in sorted_params.items())
        signature = self._make_signature("GET", path, query_str, timestamp)
        headers = {"X-API-KEY": self.api_key, "X-TIMESTAMP": timestamp, "X-SIGNATURE": signature, "Accept": "application/json"}
        try:
            response = requests.get(f"{self.base_url}/{path}", params=sorted_params, headers=headers)
            return response.json() if response.status_code == 200 else None
        except: return None

    def call_post(self, path, data):
        timestamp = str(int(time.time()))
        query_str = "" 
        signature = self._make_signature("POST", path, query_str, timestamp)
        headers = {"X-API-KEY": self.api_key, "X-TIMESTAMP": timestamp, "X-SIGNATURE": signature, "Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
        url = f"{self.base_url}/{path}"
        try:
            response = requests.post(url, data=data, headers=headers)
            return response.json() if response.status_code == 200 else None
        except: return None

api_client = StatizAPI(API_KEY, SECRET)

# 팀 코드와 실제 한국어 팀명 간의 매핑 테이블
TEAM_NAME_MAP = {
    1001: '삼성', 2002: 'KIA', 3001: '롯데', 5002: 'LG', 6002: '두산',
    7002: '한화', 9002: 'SSG', 10001: '키움', 11001: 'NC', 12001: 'KT',
    4003: '키움', 8005: '롯데'
}

def get_team_name(t_code, default_name=None):
    """
    팀 코드를 바탕으로 한글 팀명을 안전하게 조회하는 유틸리티
    """
    if default_name and isinstance(default_name, str) and not default_name.isdigit() and len(default_name) > 0:
        return default_name
    try:
        code_int = int(float(t_code))
        return TEAM_NAME_MAP.get(code_int, str(t_code))
    except: return str(t_code)

def get_now(): 
    return datetime.now().strftime('%H:%M:%S')

def calculate_win_prob(h_mu, a_mu):
    """
    Skellam 분포를 통한 최종 승률 산출
    """
    mu1, mu2 = max(h_mu, 0.01), max(a_mu, 0.01)
    win_p = 1 - skellam.cdf(0, mu1, mu2)
    loss_p = skellam.cdf(-1, mu1, mu2)
    total = win_p + loss_p
    return win_p / total if total > 0 else 0.5

# ---------------------------------------------------------
# 🛡️ 최적화 파라미터 로드 시스템 (Option A: 5종 앙상블)
# ---------------------------------------------------------
PARAMS_FILE = "best_hyperparameters.csv"
DEFAULT_PARAMS = {
    "w_2023": 1.26, "w_2024": 0.63, "w_2025": 0.68, "w_2026": 0.90,
    "cb_depth": 6, "cb_l2": 9.56, "cb_lr": 0.08,
    "rf_depth": 10, "rf_mss": 5,
    "svr_c": 1.0, "svr_epsilon": 0.1, "svr_gamma": "scale",
    "lgb_leaves": 30, "lgb_l1": 7.84, "lgb_l2": 1.20, "lgb_lr": 0.016,
    "ridge_alpha": 28.66,
    "ens_cb": 0.40, "ens_rf": 0.25, "ens_svr": 0.10, "ens_lgb": 0.15, "ens_rd": 0.10
}

def load_optimized_params():
    if not os.path.exists(PARAMS_FILE):
        return DEFAULT_PARAMS
    try:
        best_df = pd.read_csv(PARAMS_FILE)
        return best_df.iloc[0].to_dict()
    except:
        return DEFAULT_PARAMS

def run_realtime_prediction_system():
    """
    실시간 KBO 경기 일정 및 라인업을 감시하여 승패 확률을 예측하고 API로 전송하는 메인 데몬 루프
    """
    console.clear()
    console.print(Panel.fit("[bold cyan]🚀 KBO Infinite Watchdog v10.0 (5종 앙상블 고도화)[/bold cyan]", border_style="cyan"))
    
    P = load_optimized_params()
    today_str = datetime.now().strftime('%Y%m%d')
    year, month, day_key = today_str[:4], today_str[4:6], today_str[4:8]

    # 1. 5종 앙상블 모델 사전 학습
    with Status("[bold magenta]🧠 5종 앙상블 모델 최적화 학습 중...", console=console) as status:
        try:
            train_df = pd.read_csv("data/final/final_training_set_v8.csv")
            train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            train_df.fillna(0, inplace=True)
            
            weight_map = {2023: P['w_2023'], 2024: P['w_2024'], 2025: P['w_2025'], 2026: P['w_2026']}
            train_df['sample_weight'] = train_df['year'].map(weight_map)
            
            all_teams = sorted(set(train_df['homeTeam'].unique()) | set(train_df['awayTeam'].unique()))
            team_map = {team: i for i, team in enumerate(all_teams)}
            for col in ['homeTeam', 'awayTeam']: 
                train_df[col] = train_df[col].map(team_map)

            cat_features = ['homeTeam', 'awayTeam']
            features = [col for col in train_df.columns if col not in ['s_no', 'year', 'homeScore', 'awayScore', 'sample_weight']]
            
            # Ridge/SVR 전용 데이터 전처리 및 스케일러 빌드
            X_tr_ridge_raw = pd.get_dummies(train_df[features], columns=cat_features).fillna(0)
            ridge_cols = X_tr_ridge_raw.columns.tolist()
            scaler = StandardScaler()
            X_tr_ridge_scaled = np.nan_to_num(scaler.fit_transform(X_tr_ridge_raw))

            X_tr_rf = X_tr_ridge_raw

            ensemble_models = {'homeScore': {}, 'awayScore': {}}
            for target in ['homeScore', 'awayScore']:
                y_t, sw = train_df[target], train_df['sample_weight']
                
                # 1. CatBoost
                ensemble_models[target]['cb'] = CatBoostRegressor(
                    iterations=500, learning_rate=P['cb_lr'], depth=int(P['cb_depth']), 
                    l2_leaf_reg=P['cb_l2'], random_strength=P.get('cb_rs', 1.0), bagging_temperature=P.get('cb_bt', 0.0),
                    loss_function='Poisson', verbose=0
                ).fit(train_df[features], y_t, sample_weight=sw, cat_features=cat_features)
                
                # 2. RandomForest
                ensemble_models[target]['rf'] = RandomForestRegressor(
                    n_estimators=200, max_depth=int(P['rf_depth']), min_samples_split=int(P['rf_mss']),
                    random_state=42, n_jobs=-1
                ).fit(X_tr_rf, y_t, sample_weight=sw)
                
                # 3. LightGBM
                ensemble_models[target]['lgb'] = LGBMRegressor(
                    n_estimators=500, learning_rate=P['lgb_lr'], num_leaves=int(P['lgb_leaves']),
                    reg_lambda=P['lgb_l2'], reg_alpha=P['lgb_l1'], objective='poisson', verbose=-1
                ).fit(train_df[features], y_t, sample_weight=sw, categorical_feature=cat_features)

                # 4. SVR
                ensemble_models[target]['svr'] = SVR(
                    C=P['svr_c'], epsilon=P['svr_epsilon'], gamma=P['svr_gamma'], kernel='rbf'
                ).fit(X_tr_ridge_scaled, y_t, sample_weight=sw)

                # 5. Ridge
                ensemble_models[target]['ridge'] = Ridge(alpha=P['ridge_alpha']).fit(X_tr_ridge_scaled, y_t, sample_weight=sw)
            
            console.print("[bold green]✅ 모델 학습 및 실전 배치 완료[/bold green]")
        except Exception as e:
            console.print(f"[bold red]❌ 초기화 오류: {e}[/bold red]"); return

    # 기초 지표 데이터 로드
    rosters = pd.read_csv("data/raw/rosters.csv")
    day = pd.read_csv("data/processed/player_day_processed.csv")
    season = pd.read_csv("data/processed/player_season_processed.csv")
    p_hybrid = calculate_pitcher_hybrid_stats(day, season)
    b_hybrid = calculate_batter_hybrid_stats(day, season)
    rp_rolling = calculate_team_bullpen_rolling(day)

    # 가중치 정규화
    total_w = P['ens_cb'] + P['ens_rf'] + P['ens_lgb'] + P['ens_svr'] + P['ens_rd']
    w_cb, w_rf, w_lgb, w_svr, w_rd = P['ens_cb']/total_w, P['ens_rf']/total_w, P['ens_lgb']/total_w, P['ens_svr']/total_w, P['ens_rd']/total_w

    processed_s_nos = set()
    lineup_dir = "data/raw/lineups_daily"
    os.makedirs(lineup_dir, exist_ok=True)

    # 대시보드 테이블 UI
    results_table = Table(header_style="bold magenta", box=None, expand=True)
    results_table.add_column("Time", width=12)
    results_table.add_column("Matchup")
    results_table.add_column("Win Prob", justify="right")
    results_table.add_column("Prediction", justify="center")
    results_table.add_column("API Status", justify="center")

    with Live(Panel(results_table, title=f"📡 {today_str} 라인업 실시간 감시 중...", border_style="blue"), refresh_per_second=1) as live:
        while True:
            schedule = api_client.call("prediction/gameSchedule", {"year": year, "month": month})
            if not schedule or day_key not in schedule:
                time.sleep(300); continue
            
            today_games = schedule[day_key]
            pending_games = [g for g in today_games if int(g['s_no']) not in processed_s_nos]
            
            if not pending_games and len(processed_s_nos) >= len(today_games):
                live.update(Panel(results_table, title="🏁 오늘 모든 경기 처리 완료", border_style="gold1"))
                break

            for g in pending_games:
                s_no = int(g['s_no'])
                l_data = api_client.call("prediction/gameLineup", {"s_no": s_no})
                if not l_data: continue

                all_players = []
                if isinstance(l_data, list): 
                    all_players = l_data
                elif isinstance(l_data, dict):
                    for v in l_data.values():
                        if isinstance(v, list): 
                            all_players.extend(v)
                
                if not all_players: continue
                home_l = [p for p in all_players if str(p.get('t_code')) == str(g.get('homeTeam'))]
                away_l = [p for p in all_players if str(p.get('t_code')) == str(g.get('awayTeam'))]
                if len(home_l) < 9 or len(away_l) < 9: continue

                h_name, a_name = get_team_name(g.get('homeTeam'), g.get('homeTeamName')), get_team_name(g.get('awayTeam'), g.get('awayTeamName'))
                
                lineup_df = pd.DataFrame(all_players)
                lineup_df['s_no'] = s_no
                X_feat = merge_features_v7(pd.DataFrame([g]), lineup_df, rosters, p_hybrid, b_hybrid, rp_rolling, is_prediction=True)
                
                if not X_feat.empty:
                    X_test = X_feat.drop(columns=['s_no', 'year'], errors='ignore')
                    for col in ['homeTeam', 'awayTeam']: 
                        X_test[col] = X_test[col].map(team_map)
                    
                    X_test_ridge = pd.get_dummies(X_test, columns=cat_features).reindex(columns=ridge_cols, fill_value=0).replace([np.inf, -np.inf], np.nan).fillna(0)
                    X_test_ridge_sc = np.nan_to_num(scaler.transform(X_test_ridge))
                    X_test_rf = X_test_ridge

                    # 5종 앙상블을 통한 홈/어웨이 예상 득점 도출
                    mu_h = (ensemble_models['homeScore']['cb'].predict(X_test)[0] * w_cb) + \
                           (ensemble_models['homeScore']['rf'].predict(X_test_rf)[0] * w_rf) + \
                           (ensemble_models['homeScore']['lgb'].predict(X_test)[0] * w_lgb) + \
                           (ensemble_models['homeScore']['svr'].predict(X_test_ridge_sc)[0] * w_svr) + \
                           (ensemble_models['homeScore']['ridge'].predict(X_test_ridge_sc)[0] * w_rd)
                           
                    mu_a = (ensemble_models['awayScore']['cb'].predict(X_test)[0] * w_cb) + \
                           (ensemble_models['awayScore']['rf'].predict(X_test_rf)[0] * w_rf) + \
                           (ensemble_models['awayScore']['lgb'].predict(X_test)[0] * w_lgb) + \
                           (ensemble_models['awayScore']['svr'].predict(X_test_ridge_sc)[0] * w_svr) + \
                           (ensemble_models['awayScore']['ridge'].predict(X_test_ridge_sc)[0] * w_rd)
                    
                    prob = calculate_win_prob(mu_h, mu_a)
                    win_team = h_name if prob >= 0.5 else a_name
                    
                    # API 전송 payload 구성
                    payload = {
                        "ptt_idx": PTT_IDX, 
                        "s_no": s_no, 
                        "homeTeam": h_name, 
                        "awayTeam": a_name, 
                        "predictWinTeam": win_team, 
                        "percent": round(float(prob) * 100, 2), 
                        "update_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    res = api_client.call_post("prediction/savePrediction", payload)
                    api_status = "[bold green]Success[/bold green]" if res and res.get('result_cd') == 100 else "[bold red]Fail[/bold red]"
                    
                    results_table.add_row(get_now(), f"{a_name} vs {h_name}", f"{prob:.1%}", f"[bold yellow]{win_team}[/bold yellow]", api_status)
                    processed_s_nos.add(s_no)

            time.sleep(60)

if __name__ == "__main__":
    run_realtime_prediction_system()
