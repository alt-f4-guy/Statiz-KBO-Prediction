import os
import optuna
import pandas as pd
import numpy as np
import warnings
import json
import gc
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import skellam
from sklearn.metrics import accuracy_score
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout

warnings.filterwarnings('ignore')
console = Console()

# ---------------------------------------------------------
# 📂 [최적화] 전역 데이터 사전 처리
# ---------------------------------------------------------
TRAIN_FILE = "data/final/final_training_set_v8.csv"
DF_GLOBAL = pd.read_csv(TRAIN_FILE)
DF_GLOBAL.replace([np.inf, -np.inf], np.nan, inplace=True)
DF_GLOBAL.fillna(0, inplace=True)
DF_GLOBAL['game_date'] = DF_GLOBAL['s_no'].astype(str).str[:8]

ALL_TEAMS = sorted(set(DF_GLOBAL['homeTeam'].unique()) | set(DF_GLOBAL['awayTeam'].unique()))
TEAM_MAP = {team: i for i, team in enumerate(ALL_TEAMS)}
for col in ['homeTeam', 'awayTeam']:
    DF_GLOBAL[col] = DF_GLOBAL[col].map(TEAM_MAP)

CAT_FEATURES = ['homeTeam', 'awayTeam']
FEATURES = [col for col in DF_GLOBAL.columns if col not in ['s_no', 'year', 'homeScore', 'awayScore', 'game_date', 'sample_weight']]

# 전처리 사전 수행
DF_RIDGE_ALL = pd.get_dummies(DF_GLOBAL[FEATURES], columns=CAT_FEATURES).fillna(0)
RIDGE_COLS = DF_RIDGE_ALL.columns.tolist()
DF_CAT_MODELS = DF_GLOBAL[FEATURES].copy()
for col in CAT_FEATURES:
    DF_CAT_MODELS[col] = DF_CAT_MODELS[col].astype('category')

def calculate_win_prob(h_mu, a_mu):
    mu1, mu2 = max(h_mu, 0.01), max(a_mu, 0.01)
    win_p = 1 - skellam.cdf(0, mu1, mu2)
    loss_p = skellam.cdf(-1, mu1, mu2)
    total = win_p + loss_p
    return win_p / total if total > 0 else 0.5

# ---------------------------------------------------------
# 🤖 Optuna Objective 함수
# ---------------------------------------------------------

def objective(trial):
    weight_map = {
        2023: trial.suggest_float("w_2023", 0.5, 1.5),
        2024: trial.suggest_float("w_2024", 0.5, 1.5),
        2025: trial.suggest_float("w_2025", 0.5, 1.5),
        2026: trial.suggest_float("w_2026", 0.5, 1.5)
    }
    
    cb_params = {
        "iterations": 500,
        "depth": trial.suggest_int("cb_depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("cb_l2", 10.0, 50.0),
        "learning_rate": trial.suggest_float("cb_lr", 0.01, 0.1),
        "random_strength": trial.suggest_float("cb_rs", 2.0, 15.0),
        "bagging_temperature": trial.suggest_float("cb_bt", 0.0, 1.0),
        "loss_function": "Poisson", "verbose": 0, "thread_count": 6,
        "random_seed": 42
    }
    
    rf_params = {
        "n_estimators": 200,
        "max_depth": trial.suggest_int("rf_depth", 5, 15),
        "min_samples_split": trial.suggest_int("rf_mss", 2, 10),
        "random_state": 42,
        "n_jobs": 6
    }
    
    svr_params = {
        "C": trial.suggest_float("svr_c", 0.1, 10.0),
        "epsilon": trial.suggest_float("svr_epsilon", 0.01, 0.5),
        "gamma": trial.suggest_categorical("svr_gamma", ["scale", "auto"]),
        "kernel": "rbf"
    }

    lgb_params = {
        "n_estimators": 500,
        "num_leaves": trial.suggest_int("lgb_leaves", 15, 63),
        "lambda_l1": trial.suggest_float("lgb_l1", 0.0, 20.0),
        "lambda_l2": trial.suggest_float("lgb_l2", 0.0, 20.0),
        "learning_rate": trial.suggest_float("lgb_lr", 0.01, 0.1),
        "objective": "poisson", "verbose": -1, "n_jobs": 6,
        "random_state": 42
    }
    
    ridge_alpha = trial.suggest_float("ridge_alpha", 1.0, 100.0)

    w_cb = trial.suggest_float("ens_cb", 0.3, 0.8)
    w_rf = trial.suggest_float("ens_rf", 0.1, 0.4)
    w_lgb = trial.suggest_float("ens_lgb", 0.1, 0.4)
    w_svr = trial.suggest_float("ens_svr", 0.05, 0.3)
    w_rd = trial.suggest_float("ens_rd", 0.0, 0.2)

    df_2026 = DF_GLOBAL[DF_GLOBAL['year'] == 2026].copy()
    if df_2026.empty: return 0.0
    
    df_2026['month'] = df_2026['game_date'].str[4:6]
    months = sorted(df_2026['month'].unique())
    
    total_correct = 0
    total_valid_games = 0

    for m in months:
        test_idx = df_2026[df_2026['month'] == m].index
        target_month_start = df_2026[df_2026['month'] == m]['game_date'].min()
        train_idx = DF_GLOBAL.index[DF_GLOBAL['game_date'] < target_month_start]
        
        sw = DF_GLOBAL.loc[train_idx, 'year'].map(weight_map)
        tw = w_cb + w_rf + w_lgb + w_svr + w_rd
        nw = (w_cb/tw, w_rf/tw, w_lgb/tw, w_svr/tw, w_rd/tw)
        
        scaler = StandardScaler()
        X_tr_rd = np.nan_to_num(scaler.fit_transform(DF_RIDGE_ALL.loc[train_idx]))
        X_ts_rd = np.nan_to_num(scaler.transform(DF_RIDGE_ALL.loc[test_idx]))
        
        X_tr_rf = DF_RIDGE_ALL.loc[train_idx].fillna(0)
        X_ts_rf = DF_RIDGE_ALL.loc[test_idx].fillna(0)

        mu_preds = {'home': np.zeros(len(test_idx)), 'away': np.zeros(len(test_idx))}

        for target in ['homeScore', 'awayScore']:
            y_tr = DF_GLOBAL.loc[train_idx, target]
            m_cb = CatBoostRegressor(**cb_params).fit(DF_GLOBAL.loc[train_idx, FEATURES], y_tr, sample_weight=sw, cat_features=CAT_FEATURES)
            p_cb = m_cb.predict(DF_GLOBAL.loc[test_idx, FEATURES])
            
            m_rf = RandomForestRegressor(**rf_params).fit(X_tr_rf, y_tr, sample_weight=sw)
            p_rf = m_rf.predict(X_ts_rf)
            
            m_lgb = LGBMRegressor(**lgb_params).fit(DF_CAT_MODELS.loc[train_idx], y_tr, sample_weight=sw)
            p_lgb = m_lgb.predict(DF_CAT_MODELS.loc[test_idx])
            
            m_svr = SVR(**svr_params).fit(X_tr_rd, y_tr, sample_weight=sw)
            p_svr = m_svr.predict(X_ts_rd)
            
            m_rd = Ridge(alpha=ridge_alpha, random_state=42).fit(X_tr_rd, y_tr, sample_weight=sw)
            p_rd = m_rd.predict(X_ts_rd)

            key = 'home' if target == 'homeScore' else 'away'
            mu_preds[key] = (p_cb * nw[0]) + (p_rf * nw[1]) + (p_lgb * nw[2]) + (p_svr * nw[3]) + (p_rd * nw[4])
            del m_cb, m_rf, m_lgb, m_svr, m_rd

        for i, row_idx in enumerate(test_idx):
            actual_row = DF_GLOBAL.loc[row_idx]
            prob = calculate_win_prob(mu_preds['home'][i], mu_preds['away'][i])
            actual = "Home" if actual_row['homeScore'] > actual_row['awayScore'] else ("Away" if actual_row['homeScore'] < actual_row['awayScore'] else "Draw")
            if actual != "Draw":
                pred = "Home" if prob >= 0.5 else "Away"
                if pred == actual: total_correct += 1
                total_valid_games += 1

    gc.collect()
    return total_correct / total_valid_games if total_valid_games > 0 else 0.0

def make_dashboard(study):
    table = Table(title="[bold cyan]Optuna Hyperparameter Tuning[/bold cyan]", show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Trial", justify="center")
    table.add_column("Value (Acc)", justify="right")
    table.add_column("Best Value", justify="right", style="bold green")
    
    best_val = study.best_value if len(study.trials) > 0 else 0
    current_trial = len(study.trials)
    
    last_val = study.trials[-1].value if len(study.trials) > 0 else 0
    table.add_row(str(current_trial), f"{last_val:.4f}", f"{best_val:.4f}")
    
    param_table = Table(title="[bold yellow]Best Parameters[/bold yellow]", show_header=True, header_style="bold yellow", expand=True)
    param_table.add_column("Parameter")
    param_table.add_column("Value")
    
    if len(study.trials) > 0:
        for k, v in study.best_params.items():
            param_table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
            
    return Panel(Layout(table), title=f"🚀 Tuning in Progress... (Trial {current_trial})", border_style="blue"), Panel(param_table, border_style="yellow")

if __name__ == "__main__":
    N_TRIALS = 300
    console.print(Panel.fit("[bold cyan]🚀 KBO All-In Hyperparameter Optimizer[/bold cyan]", border_style="cyan"))
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    
    with Live(console=console, refresh_per_second=4) as live:
        def callback(study, trial):
            dash1, dash2 = make_dashboard(study)
            # Layout 사용 대신 간단하게 print용으로 구성
            combined = Table.grid(expand=True)
            combined.add_column()
            combined.add_column()
            combined.add_row(dash1, dash2)
            live.update(combined)

        study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback])

    console.print("\n" + "="*50)
    console.print(f"🏆 [bold green]최적화 완료 (최고 정확도: {study.best_value:.4f})[/bold green]")
    
    pd.DataFrame([study.best_params]).to_csv("Option_A/best_hyperparameters.csv", index=False)
    console.print("\n📝 [bold yellow]최적 파라미터 저장 완료: Option_A/best_hyperparameters.csv[/bold yellow]")
