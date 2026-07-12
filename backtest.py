import os
import pandas as pd
import numpy as np
import warnings
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

# 경고 무시 처리
warnings.filterwarnings('ignore')
console = Console()

# ---------------------------------------------------------
# 📂 설정 및 경로
# ---------------------------------------------------------
TRAIN_FILE = "data/final/final_training_set_v8.csv"
PARAMS_FILE = "best_hyperparameters.csv"
TEST_YEAR = 2026

# 🛡️ 기본 파라미터 (파일이 없을 경우 사용될 백업용 파라미터)
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
    """
    최적 튜닝 하이퍼파라미터 파일(best_hyperparameters.csv) 로드 함수
    """
    if not os.path.exists(PARAMS_FILE):
        return DEFAULT_PARAMS
    try:
        best_df = pd.read_csv(PARAMS_FILE)
        return best_df.iloc[0].to_dict()
    except:
        return DEFAULT_PARAMS

def calculate_win_prob(h_mu, a_mu):
    """
    Skellam 분포를 사용하여 두 팀의 득점 분포 차이를 도출하고 승률 계산
    """
    mu1, mu2 = max(h_mu, 0.01), max(a_mu, 0.01)
    win_p = 1 - skellam.cdf(0, mu1, mu2)
    loss_p = skellam.cdf(-1, mu1, mu2)
    total = win_p + loss_p
    return win_p / total if total > 0 else 0.5

def run_backtest():
    """
    2026 시즌을 대상으로 Walk-forward 방식의 예측 성능 백테스트 실행
    """
    console.clear()
    console.print(Panel.fit("[bold cyan]⚾ KBO Backtest System v2.0[/bold cyan]", border_style="cyan"))
    
    P = load_optimized_params()
    
    if not os.path.exists(TRAIN_FILE):
        console.print(f"[bold red]❌ 파일을 찾을 수 없습니다: {TRAIN_FILE}[/bold red]"); return
    
    # 학습 세트 로드 및 무한대/결측치 정형화
    df = pd.read_csv(TRAIN_FILE)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    # 연도별 샘플 가중치 매핑
    weight_map = {2023: P['w_2023'], 2024: P['w_2024'], 2025: P['w_2025'], 2026: P['w_2026']}
    df['sample_weight'] = df['year'].map(weight_map)
    df['game_date'] = df['s_no'].astype(str).str[:8]
    
    all_teams = sorted(set(df['homeTeam'].unique()) | set(df['awayTeam'].unique()))
    team_map = {team: i for i, team in enumerate(all_teams)}
    df_orig = df.copy()
    for col in ['homeTeam', 'awayTeam']: 
        df[col] = df[col].map(team_map)
    
    cat_features = ['homeTeam', 'awayTeam']
    features = [col for col in df.columns if col not in ['s_no', 'year', 'homeScore', 'awayScore', 'game_date', 'sample_weight']]
    
    # 2026 시즌 테스트 대상 필터링 및 날짜순 정렬
    test_data = df[df['year'] == TEST_YEAR].sort_values('s_no')
    chunk_size = 5
    num_chunks = int(np.ceil(len(test_data) / chunk_size))
    
    all_results = []
    
    # 앙상블 블렌딩 가중치 정규화
    total_ens_w = P['ens_cb'] + P['ens_rf'] + P['ens_lgb'] + P['ens_svr'] + P['ens_rd']
    w_cb, w_rf, w_lgb, w_svr, w_rd = P['ens_cb']/total_ens_w, P['ens_rf']/total_ens_w, P['ens_lgb']/total_ens_w, P['ens_svr']/total_ens_w, P['ens_rd']/total_ens_w

    # --- UI 구성을 위한 rich 프로그레스바 ---
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    )
    overall_task = progress.add_task("[yellow]전체 시즌 분석 중...", total=num_chunks)
    
    display_results = [] # 최근 경기 결과를 보여주는 버퍼

    def make_layout(acc: float, correct: int, total: int, table: Table):
        layout = Layout()
        layout.split_column(
            Layout(Panel(f"[bold green]📊 실시간 적중률: {acc:.2f}% ({correct}/{total})[/bold green]", border_style="green"), size=3),
            Layout(Panel(table, title="최근 경기 결과", border_style="blue")),
            Layout(progress, size=3)
        )
        return layout

    # 초기 빈 테이블 생성
    results_table = Table(header_style="bold magenta", box=None, expand=True)
    results_table.add_column("Date", width=10); results_table.add_column("Matchup"); results_table.add_column("Prob", justify="right")
    results_table.add_column("Pred", justify="center"); results_table.add_column("Actual", justify="center"); results_table.add_column("Result", justify="center")

    with Live(make_layout(0, 0, 0, results_table), refresh_per_second=4) as live:
        for i in range(num_chunks):
            test_pool = test_data.iloc[i*chunk_size : (i+1)*chunk_size].copy()
            if test_pool.empty:
                continue
                
            first_s_no = test_pool['s_no'].iloc[0]
            target_date = str(first_s_no)[:8]
            
            # 테스트 시점 직전까지의 경기 데이터를 학습 데이터로 사용 (Walk-forward)
            train_pool = df[df['s_no'] < first_s_no].copy()
            if train_pool.empty:
                progress.update(overall_task, advance=1)
                continue

            progress.update(overall_task, description=f"[cyan]📅 {target_date} 모델 학습 중... ({i+1}/{num_chunks})")
            
            train_pool = train_pool.sort_values('s_no')
            train_df = train_pool
            
            ensemble_models = {'homeScore': {}, 'awayScore': {}}
            
            # Ridge/SVR 전용 원-핫 인코딩 및 스케일러 빌드
            X_tr_ridge_raw = pd.get_dummies(train_df[features], columns=cat_features).replace([np.inf, -np.inf], np.nan).fillna(0)
            ridge_cols = X_tr_ridge_raw.columns.tolist()
            scaler = StandardScaler()
            X_tr_ridge_scaled = np.nan_to_num(scaler.fit_transform(X_tr_ridge_raw))

            # RandomForest용 결측 정형화 피처 정의
            X_tr_rf = X_tr_ridge_raw
            
            for target in ['homeScore', 'awayScore']:
                y_train, w_train = train_df[target], train_df['sample_weight']
                
                # 1. CatBoostRegressor 학습
                ensemble_models[target]['cb'] = CatBoostRegressor(
                    iterations=500, learning_rate=P['cb_lr'], depth=int(P['cb_depth']), 
                    l2_leaf_reg=P['cb_l2'], random_strength=P.get('cb_rs', 1.0), bagging_temperature=P.get('cb_bt', 0.0),
                    loss_function='Poisson', verbose=0, random_seed=42
                ).fit(train_df[features], y_train, sample_weight=w_train, cat_features=cat_features)
                
                # 2. RandomForestRegressor 학습
                ensemble_models[target]['rf'] = RandomForestRegressor(
                    n_estimators=200, max_depth=int(P['rf_depth']), min_samples_split=int(P['rf_mss']),
                    random_state=42, n_jobs=-1
                ).fit(X_tr_rf, y_train, sample_weight=w_train)
                
                # 3. LGBMRegressor 학습
                ensemble_models[target]['lgb'] = LGBMRegressor(
                    n_estimators=500, learning_rate=P['lgb_lr'], num_leaves=int(P['lgb_leaves']),
                    reg_lambda=P['lgb_l2'], reg_alpha=P['lgb_l1'], objective='poisson', verbose=-1, random_state=42
                ).fit(train_df[features], y_train, sample_weight=w_train, categorical_feature=cat_features)
                
                # 4. SVR 학습
                ensemble_models[target]['svr'] = SVR(
                    C=P['svr_c'], epsilon=P['svr_epsilon'], gamma=P['svr_gamma'], kernel='rbf'
                ).fit(X_tr_ridge_scaled, y_train, sample_weight=w_train)

                # 5. Ridge 회귀 학습
                ensemble_models[target]['ridge'] = Ridge(alpha=P['ridge_alpha'], random_state=42).fit(X_tr_ridge_scaled, y_train, sample_weight=w_train)

            progress.update(overall_task, description=f"[green]🔮 {target_date} 경기 예측 중...")
            
            # 테스트 셋 피처 스케일링 및 매핑 정렬
            X_test_all = test_pool[features]
            X_test_ridge = pd.get_dummies(X_test_all, columns=cat_features).reindex(columns=ridge_cols, fill_value=0).replace([np.inf, -np.inf], np.nan).fillna(0)
            X_test_ridge_scaled = np.nan_to_num(scaler.transform(X_test_ridge))
            X_test_rf = X_test_ridge

            # 경기별 개별 예측 루프 수행
            for idx, row in test_pool.iterrows():
                pos = test_pool.index.get_loc(idx)
                game_x, game_x_rf, game_x_rd = X_test_all.loc[[idx]], X_test_rf.loc[[idx]], X_test_ridge_scaled[[pos]]
                mu_dict = {}
                for target in ['homeScore', 'awayScore']:
                    p_cb = ensemble_models[target]['cb'].predict(game_x)[0]
                    p_rf = ensemble_models[target]['rf'].predict(game_x_rf)[0]
                    p_lgb = ensemble_models[target]['lgb'].predict(game_x)[0]
                    p_svr = ensemble_models[target]['svr'].predict(game_x_rd)[0]
                    p_rd = ensemble_models[target]['ridge'].predict(game_x_rd)[0]
                    
                    # 5종 모델 가중 결합을 통해 최종 득점 분포 기댓값(mu) 계산
                    mu_dict[target] = (p_cb * w_cb) + (p_rf * w_rf) + (p_lgb * w_lgb) + (p_svr * w_svr) + (p_rd * w_rd)
                
                prob = calculate_win_prob(mu_dict['homeScore'], mu_dict['awayScore'])
                pred_winner = "Home" if prob >= 0.5 else "Away"
                actual_winner = "Home" if row['homeScore'] > row['awayScore'] else ("Away" if row['homeScore'] < row['awayScore'] else "Draw")
                is_correct = (pred_winner == actual_winner)
                
                all_results.append({'date': target_date, 'correct': is_correct, 'actual': actual_winner})
                
                res_symbol = "[bold green]⭕ 적중[/bold green]" if is_correct else "[bold red]❌ 실패[/bold red]"
                if actual_winner == "Draw": 
                    res_symbol = "[white]무승부[/white]"
                
                # UI 데이터 리스트 갱신 (최신 8개 유지)
                display_results.append([
                    target_date[4:], f"Team {int(df_orig.loc[idx, 'awayTeam'])} @ {int(df_orig.loc[idx, 'homeTeam'])}",
                    f"{prob:.1%}", pred_winner, actual_winner, res_symbol
                ])
                if len(display_results) > 8: 
                    display_results.pop(0)

                # 결과 출력용 테이블 실시간 업데이트
                results_table = Table(header_style="bold magenta", box=None, expand=True)
                results_table.add_column("Date", width=10); results_table.add_column("Matchup"); results_table.add_column("Prob", justify="right")
                results_table.add_column("Pred", justify="center"); results_table.add_column("Actual", justify="center"); results_table.add_column("Result", justify="center")
                for r in display_results: 
                    results_table.add_row(*r)

            res_temp = pd.DataFrame(all_results)
            valid_temp = res_temp[res_temp['actual'] != "Draw"]
            current_acc = (valid_temp['correct'].mean() * 100) if not valid_temp.empty else 0
            live.update(make_layout(current_acc, int(valid_temp['correct'].sum()), len(valid_temp), results_table))
            progress.update(overall_task, advance=1)

    console.print(Panel(f"[bold gold1]🏁 백테스트 완료! 최종 정확도: {current_acc:.2f}%[/bold gold1]", border_style="gold1"))
    pd.DataFrame(all_results).to_csv("backtest_results_2026.csv", index=False)

if __name__ == "__main__":
    run_backtest()
