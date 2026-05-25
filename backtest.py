import os
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import skellam
from concurrent.futures import ProcessPoolExecutor, as_completed

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
PARAMS_FILE = "best_hyperparameters.csv"
TEST_YEAR = 2026

DEFAULT_PARAMS = {
    "w_2023": 1.26, "w_2024": 0.63, "w_2025": 0.68, "w_2026": 0.90,
    "cb_depth": 6, "cb_l2": 9.56, "cb_lr": 0.08,
    "xg_depth": 4, "xg_lambda": 16.63, "xg_alpha": 3.70, "xg_lr": 0.02,
    "lgb_leaves": 30, "lgb_l1": 7.84, "lgb_l2": 1.20, "lgb_lr": 0.016,
    "ridge_alpha": 28.66,
    "ens_cb": 0.40, "ens_xg": 0.25, "ens_lgb": 0.17, "ens_rd": 0.18
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

def process_date(target_date, df, features, cat_features, P):
    train_pool = df[df['game_date'] < target_date].copy()
    test_pool = df[df['game_date'] == target_date].copy()
    if train_pool.empty or test_pool.empty:
        return []

    train_pool = train_pool.sort_values('s_no')
    val_idx = int(len(train_pool) * 0.9)
    train_df, val_df = train_pool.iloc[:val_idx], train_pool.iloc[val_idx:]
    
    ensemble_models = {'homeScore': {}, 'awayScore': {}}
    
    # Ridge 전처리
    X_tr_ridge_raw = pd.get_dummies(train_df[features], columns=cat_features).replace([np.inf, -np.inf], np.nan).fillna(0)
    ridge_cols = X_tr_ridge_raw.columns.tolist()
    scaler = StandardScaler()
    X_tr_ridge_scaled = np.nan_to_num(scaler.fit_transform(X_tr_ridge_raw))
    
    for target in ['homeScore', 'awayScore']:
        y_train, y_val, w_train = train_df[target], val_df[target], train_df['sample_weight']
        
        # 내부 스레드를 1개로 제한하여 멀티프로세스 환경에서 CPU 코어 경합을 막음
        ensemble_models[target]['cb'] = CatBoostRegressor(
            iterations=500, learning_rate=P['cb_lr'], depth=int(P['cb_depth']), 
            l2_leaf_reg=P['cb_l2'], random_strength=P.get('cb_rs', 1.0), bagging_temperature=P.get('cb_bt', 0.0),
            loss_function='Poisson', verbose=0, early_stopping_rounds=50, thread_count=1
        ).fit(train_df[features], y_train, sample_weight=w_train, eval_set=(val_df[features], y_val), cat_features=cat_features)
        
        X_tr_xg, X_val_xg = train_df[features].copy(), val_df[features].copy()
        for col in cat_features:
            X_tr_xg[col], X_val_xg[col] = X_tr_xg[col].astype('category'), X_val_xg[col].astype('category')
        
        ensemble_models[target]['xg'] = XGBRegressor(
            n_estimators=500, learning_rate=P['xg_lr'], max_depth=int(P['xg_depth']),
            reg_lambda=P['xg_lambda'], reg_alpha=P['xg_alpha'], subsample=0.8,
            objective='count:poisson', tree_method='hist', enable_categorical=True, early_stopping_rounds=50, n_jobs=1
        ).fit(X_tr_xg, y_train, sample_weight=w_train, eval_set=[(X_val_xg, y_val)], verbose=False)
        
        ensemble_models[target]['lgb'] = LGBMRegressor(
            n_estimators=500, learning_rate=P['lgb_lr'], num_leaves=int(P['lgb_leaves']),
            reg_lambda=P['lgb_l2'], reg_alpha=P['lgb_l1'], objective='poisson', verbose=-1, early_stopping_rounds=50, n_jobs=1
        ).fit(train_df[features], y_train, sample_weight=w_train, eval_set=[(val_df[features], y_val)], categorical_feature=cat_features)

        ensemble_models[target]['ridge'] = Ridge(alpha=P['ridge_alpha']).fit(X_tr_ridge_scaled, y_train, sample_weight=w_train)

    X_test_all = test_pool[features]
    X_test_ridge = pd.get_dummies(X_test_all, columns=cat_features).reindex(columns=ridge_cols, fill_value=0).replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test_ridge_scaled = np.nan_to_num(scaler.transform(X_test_ridge))
    X_test_xg = X_test_all.copy()
    for col in cat_features: X_test_xg[col] = X_test_xg[col].astype('category')

    total_ens_w = P['ens_cb'] + P['ens_xg'] + P['ens_lgb'] + P['ens_rd']
    w_cb, w_xg, w_lgb, w_rd = P['ens_cb']/total_ens_w, P['ens_xg']/total_ens_w, P['ens_lgb']/total_ens_w, P['ens_rd']/total_ens_w

    results = []
    for idx, row in test_pool.iterrows():
        pos = test_pool.index.get_loc(idx)
        game_x, game_x_xg, game_x_rd = X_test_all.loc[[idx]], X_test_xg.loc[[idx]], X_test_ridge_scaled[[pos]]
        mu_dict = {}
        for target in ['homeScore', 'awayScore']:
            p_cb = ensemble_models[target]['cb'].predict(game_x)[0]
            p_xg = ensemble_models[target]['xg'].predict(game_x_xg)[0]
            p_lgb = ensemble_models[target]['lgb'].predict(game_x)[0]
            p_rd = ensemble_models[target]['ridge'].predict(game_x_rd)[0]
            mu_dict[target] = (p_cb * w_cb) + (p_xg * w_xg) + (p_lgb * w_lgb) + (p_rd * w_rd)
        
        prob = calculate_win_prob(mu_dict['homeScore'], mu_dict['awayScore'])
        pred_winner = "Home" if prob >= 0.5 else "Away"
        actual_winner = "Home" if row['homeScore'] > row['awayScore'] else ("Away" if row['homeScore'] < row['awayScore'] else "Draw")
        is_correct = (pred_winner == actual_winner)
        
        results.append({
            'date': target_date,
            'idx': idx,
            'prob': prob,
            'pred_winner': pred_winner,
            'actual_winner': actual_winner,
            'is_correct': is_correct
        })
    return results

def run_backtest():
    console.clear()
    console.print(Panel.fit("[bold cyan]⚾ KBO Backtest System v2.0 (병렬 최적화 적용)[/bold cyan]", border_style="cyan"))
    
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
    df_orig = df.copy()
    for col in ['homeTeam', 'awayTeam']: df[col] = df[col].map(team_map)
    
    cat_features = ['homeTeam', 'awayTeam']
    features = [col for col in df.columns if col not in ['s_no', 'year', 'homeScore', 'awayScore', 'game_date', 'sample_weight']]
    test_dates = sorted(df[df['year'] == TEST_YEAR]['game_date'].unique())
    
    all_results = []
    display_results = []
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    )
    overall_task = progress.add_task("[yellow]전체 시즌 분석 중...", total=len(test_dates))
    
    def make_layout(acc: float, correct: int, total: int, table: Table):
        layout = Layout()
        layout.split_column(
            Layout(Panel(f"[bold green]📊 실시간 적중률: {acc:.2f}% ({correct}/{total})[/bold green]", border_style="green"), size=3),
            Layout(Panel(table, title="최근 경기 결과", border_style="blue")),
            Layout(progress, size=3)
        )
        return layout

    results_table = Table(header_style="bold magenta", box=None, expand=True)
    results_table.add_column("Date", width=10); results_table.add_column("Matchup"); results_table.add_column("Prob", justify="right")
    results_table.add_column("Pred", justify="center"); results_table.add_column("Actual", justify="center"); results_table.add_column("Result", justify="center")

    with Live(make_layout(0, 0, 0, results_table), refresh_per_second=4) as live:
        # Multiprocessing Pool 생성 (mac OS의 물리/논리 코어 수 고려)
        # CatBoost, XGBoost, LightGBM 내부 n_jobs=1로 제한하여 코어 경합 최소화
        max_workers = max(1, os.cpu_count() - 1)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_date, target_date, df, features, cat_features, P): target_date 
                for target_date in test_dates
            }
            
            for future in as_completed(futures):
                t_date = futures[future]
                try:
                    res_list = future.result()
                    if not res_list:
                        progress.update(overall_task, advance=1)
                        continue
                    
                    for r in res_list:
                        all_results.append({
                            'date': r['date'],
                            'correct': r['is_correct'],
                            'actual': r['actual_winner']
                        })
                        
                        is_correct = r['is_correct']
                        res_symbol = "[bold green]⭕ 적중[/bold green]" if is_correct else "[bold red]❌ 실패[/bold red]"
                        if r['actual_winner'] == "Draw": res_symbol = "[white]무승부[/white]"
                        
                        display_results.append([
                            r['date'][4:], 
                            f"Team {int(df_orig.loc[r['idx'], 'awayTeam'])} @ {int(df_orig.loc[r['idx'], 'homeTeam'])}",
                            f"{r['prob']:.1%}", 
                            r['pred_winner'], 
                            r['actual_winner'], 
                            res_symbol
                        ])
                    
                    # 결과를 정렬하고 최신 8개만 유지
                    display_results.sort(key=lambda x: x[0])
                    if len(display_results) > 8:
                        display_results = display_results[-8:]
                        
                    results_table = Table(header_style="bold magenta", box=None, expand=True)
                    results_table.add_column("Date", width=10); results_table.add_column("Matchup"); results_table.add_column("Prob", justify="right")
                    results_table.add_column("Pred", justify="center"); results_table.add_column("Actual", justify="center"); results_table.add_column("Result", justify="center")
                    for r_row in display_results: 
                        results_table.add_row(*r_row)
                        
                    res_temp = pd.DataFrame(all_results)
                    valid_temp = res_temp[res_temp['actual'] != "Draw"]
                    current_acc = (valid_temp['correct'].mean() * 100) if not valid_temp.empty else 0
                    
                    live.update(make_layout(current_acc, int(valid_temp['correct'].sum()), len(valid_temp), results_table))
                    progress.update(overall_task, advance=1, description=f"[cyan]📅 {t_date} 완료")
                except Exception as e:
                    console.print(f"[bold red]Error on date {t_date}: {e}[/bold red]")
                    progress.update(overall_task, advance=1)
                    
    console.print(Panel(f"[bold gold1]🏁 백테스트 완료! 최종 정확도: {current_acc:.2f}%[/bold gold1]", border_style="gold1"))
    pd.DataFrame(all_results).to_csv("backtest_results_2026.csv", index=False)

if __name__ == "__main__":
    run_backtest()
