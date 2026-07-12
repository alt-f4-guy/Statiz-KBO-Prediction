import os
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from tqdm import tqdm
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
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

warnings.filterwarnings('ignore')
console = Console()

# ---------------------------------------------------------
# 📂 설정 및 경로
# ---------------------------------------------------------
TRAIN_FILE = "data/final/final_training_set_v8.csv"
PARAMS_FILE = "Option_B/best_hyperparameters.csv"
TEST_YEAR = 2026

# 🛡️ 기본 파라미터 (파일이 없을 경우 사용될 백업용)
DEFAULT_PARAMS = {
    "w_2023": 1.26, "w_2024": 0.63, "w_2025": 0.68, "w_2026": 0.90,
    "cb_depth": 6, "cb_l2": 9.56, "cb_lr": 0.08,
    "rf_depth": 10, "rf_mss": 5,
    "mlp_hidden": (32, 16), "mlp_alpha": 1.0, "mlp_lr": 0.01,
    "lgb_leaves": 30, "lgb_l1": 7.84, "lgb_l2": 1.20, "lgb_lr": 0.016,
    "ridge_alpha": 28.66,
    "ens_cb": 0.40, "ens_rf": 0.25, "ens_mlp": 0.10, "ens_lgb": 0.15, "ens_rd": 0.10
}

def load_optimized_params():
    if not os.path.exists(PARAMS_FILE):
        return DEFAULT_PARAMS
    try:
        best_df = pd.read_csv(PARAMS_FILE)
        return best_df.iloc[0].to_dict()
    except:
        return DEFAULT_PARAMS

def calculate_win_prob(h_mu, a_mu):
    mu1, mu2 = max(h_mu, 0.01), max(a_mu, 0.01)
    win_p = 1 - skellam.cdf(0, mu1, mu2)
    loss_p = skellam.cdf(-1, mu1, mu2)
    total = win_p + loss_p
    return win_p / total if total > 0 else 0.5

def run_backtest():
    console.clear()
    console.print(Panel.fit("[bold cyan]⚾ KBO Backtest System v2.0[/bold cyan]", border_style="cyan"))
    
    P = load_optimized_params()
    
    if not os.path.exists(TRAIN_FILE):
        console.print(f"[bold red]❌ 파일을 찾을 수 없습니다: {TRAIN_FILE}[/bold red]"); return
    
    df = pd.read_csv(TRAIN_FILE)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    weight_map = {2023: P['w_2023'], 2024: P['w_2024'], 2025: P['w_2025'], 2026: P['w_2026']}
    df['sample_weight'] = df['year'].map(weight_map)
    df['game_date'] = df['s_no'].astype(str).str[:8]
    
    all_teams = sorted(set(df['homeTeam'].unique()) | set(df['awayTeam'].unique()))
    team_map = {team: i for i, team in enumerate(all_teams)}
    # 원본 팀 코드를 보여주기 위해 복사본 사용
    df_orig = df.copy()
    for col in ['homeTeam', 'awayTeam']: df[col] = df[col].map(team_map)
    
    cat_features = ['homeTeam', 'awayTeam']
    features = [col for col in df.columns if col not in ['s_no', 'year', 'homeScore', 'awayScore', 'game_date', 'sample_weight']]
    test_data = df[df['year'] == TEST_YEAR].sort_values('s_no')
    chunk_size = 5
    num_chunks = int(np.ceil(len(test_data) / chunk_size))
    
    all_results = []
    
    # 앙상블 가중치 정규화
    total_ens_w = P['ens_cb'] + P['ens_rf'] + P['ens_lgb'] + P['ens_mlp'] + P['ens_rd']
    w_cb, w_rf, w_lgb, w_mlp, w_rd = P['ens_cb']/total_ens_w, P['ens_rf']/total_ens_w, P['ens_lgb']/total_ens_w, P['ens_mlp']/total_ens_w, P['ens_rd']/total_ens_w

    # --- UI 구성을 위한 변수 ---
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    )
    overall_task = progress.add_task("[yellow]전체 시즌 분석 중...", total=num_chunks)
    
    display_results = [] # 최근 경기 결과 저장용 리스트

    def make_layout(acc: float, correct: int, total: int, table: Table):
        layout = Layout()
        layout.split_column(
            Layout(Panel(f"[bold green]📊 실시간 적중률: {acc:.2f}% ({correct}/{total})[/bold green]", border_style="green"), size=3),
            Layout(Panel(table, title="최근 경기 결과", border_style="blue")),
            Layout(progress, size=3)
        )
        return layout

    # 초기 테이블 생성
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
            
            train_pool = df[df['s_no'] < first_s_no].copy()
            if train_pool.empty:
                progress.update(overall_task, advance=1)
                continue

            progress.update(overall_task, description=f"[cyan]📅 {target_date} 모델 학습 중... ({i+1}/{num_chunks})")
            
            train_pool = train_pool.sort_values('s_no')
            train_df = train_pool
            
            ensemble_models = {'homeScore': {}, 'awayScore': {}}
            
            X_tr_ridge_raw = pd.get_dummies(train_df[features], columns=cat_features).replace([np.inf, -np.inf], np.nan).fillna(0)
            ridge_cols = X_tr_ridge_raw.columns.tolist()
            scaler = StandardScaler()
            X_tr_ridge_scaled = np.nan_to_num(scaler.fit_transform(X_tr_ridge_raw))

            X_tr_rf = X_tr_ridge_raw
            for target in ['homeScore', 'awayScore']:
                y_train, w_train = train_df[target], train_df['sample_weight']
                
                ensemble_models[target]['cb'] = CatBoostRegressor(
                    iterations=500, learning_rate=P['cb_lr'], depth=int(P['cb_depth']), 
                    l2_leaf_reg=P['cb_l2'], random_strength=P.get('cb_rs', 1.0), bagging_temperature=P.get('cb_bt', 0.0),
                    loss_function='Poisson', verbose=0, random_seed=42
                ).fit(train_df[features], y_train, sample_weight=w_train, cat_features=cat_features)
                
                ensemble_models[target]['rf'] = RandomForestRegressor(
                    n_estimators=200, max_depth=int(P['rf_depth']), min_samples_split=int(P['rf_mss']),
                    random_state=42, n_jobs=-1
                ).fit(X_tr_rf, y_train, sample_weight=w_train)
                
                ensemble_models[target]['lgb'] = LGBMRegressor(
                    n_estimators=500, learning_rate=P['lgb_lr'], num_leaves=int(P['lgb_leaves']),
                    reg_lambda=P['lgb_l2'], reg_alpha=P['lgb_l1'], objective='poisson', verbose=-1, random_state=42
                ).fit(train_df[features], y_train, sample_weight=w_train, categorical_feature=cat_features)
                
                ensemble_models[target]['mlp'] = MLPRegressor(
                    hidden_layer_sizes=eval(P['mlp_hidden']) if isinstance(P['mlp_hidden'], str) else P['mlp_hidden'], 
                    alpha=P['mlp_alpha'], learning_rate_init=P['mlp_lr'], max_iter=500, early_stopping=True, random_state=42
                ).fit(X_tr_ridge_scaled, y_train)

                ensemble_models[target]['ridge'] = Ridge(alpha=P['ridge_alpha'], random_state=42).fit(X_tr_ridge_scaled, y_train, sample_weight=w_train)

            progress.update(overall_task, description=f"[green]🔮 {target_date} 경기 예측 중...")
            
            X_test_all = test_pool[features]
            X_test_ridge = pd.get_dummies(X_test_all, columns=cat_features).reindex(columns=ridge_cols, fill_value=0).replace([np.inf, -np.inf], np.nan).fillna(0)
            X_test_ridge_scaled = np.nan_to_num(scaler.transform(X_test_ridge))
            X_test_rf = X_test_ridge

            for idx, row in test_pool.iterrows():
                pos = test_pool.index.get_loc(idx)
                game_x, game_x_rf, game_x_rd = X_test_all.loc[[idx]], X_test_rf.loc[[idx]], X_test_ridge_scaled[[pos]]
                mu_dict = {}
                for target in ['homeScore', 'awayScore']:
                    p_cb = ensemble_models[target]['cb'].predict(game_x)[0]
                    p_rf = ensemble_models[target]['rf'].predict(game_x_rf)[0]
                    p_lgb = ensemble_models[target]['lgb'].predict(game_x)[0]
                    p_mlp = ensemble_models[target]['mlp'].predict(game_x_rd)[0]
                    p_rd = ensemble_models[target]['ridge'].predict(game_x_rd)[0]
                    mu_dict[target] = (p_cb * w_cb) + (p_rf * w_rf) + (p_lgb * w_lgb) + (p_mlp * w_mlp) + (p_rd * w_rd)
                
                prob = calculate_win_prob(mu_dict['homeScore'], mu_dict['awayScore'])
                pred_winner = "Home" if prob >= 0.5 else "Away"
                actual_winner = "Home" if row['homeScore'] > row['awayScore'] else ("Away" if row['homeScore'] < row['awayScore'] else "Draw")
                is_correct = (pred_winner == actual_winner)
                
                all_results.append({'date': target_date, 'correct': is_correct, 'actual': actual_winner})
                
                res_symbol = "[bold green]⭕ 적중[/bold green]" if is_correct else "[bold red]❌ 실패[/bold red]"
                if actual_winner == "Draw": res_symbol = "[white]무승부[/white]"
                
                # UI용 데이터 리스트 업데이트
                display_results.append([
                    target_date[4:], f"Team {int(df_orig.loc[idx, 'awayTeam'])} @ {int(df_orig.loc[idx, 'homeTeam'])}",
                    f"{prob:.1%}", pred_winner, actual_winner, res_symbol
                ])
                if len(display_results) > 8: display_results.pop(0)

                # 테이블 새로 생성
                results_table = Table(header_style="bold magenta", box=None, expand=True)
                results_table.add_column("Date", width=10); results_table.add_column("Matchup"); results_table.add_column("Prob", justify="right")
                results_table.add_column("Pred", justify="center"); results_table.add_column("Actual", justify="center"); results_table.add_column("Result", justify="center")
                for r in display_results: results_table.add_row(*r)

            res_temp = pd.DataFrame(all_results)
            valid_temp = res_temp[res_temp['actual'] != "Draw"]
            current_acc = (valid_temp['correct'].mean() * 100) if not valid_temp.empty else 0
            live.update(make_layout(current_acc, int(valid_temp['correct'].sum()), len(valid_temp), results_table))
            progress.update(overall_task, advance=1)

    console.print(Panel(f"[bold gold1]🏁 백테스트 완료! 최종 정확도: {current_acc:.2f}%[/bold gold1]", border_style="gold1"))
    pd.DataFrame(all_results).to_csv("Option_B/backtest_results_2026.csv", index=False)

if __name__ == "__main__":
    run_backtest()
