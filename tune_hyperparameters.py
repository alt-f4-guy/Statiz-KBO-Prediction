import os
import optuna
import pandas as pd
import numpy as np
import warnings
import gc
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import skellam
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

warnings.filterwarnings('ignore')
console = Console()

# ---------------------------------------------------------
# 📂 설정 및 경로
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
        "iterations": 250,
        "depth": trial.suggest_int("cb_depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("cb_l2", 10.0, 50.0),
        "learning_rate": trial.suggest_float("cb_lr", 0.01, 0.1),
        "random_strength": trial.suggest_float("cb_rs", 2.0, 15.0),
        "bagging_temperature": trial.suggest_float("cb_bt", 0.0, 1.0),
        "loss_function": "Poisson", "verbose": 0, "thread_count": 1
    }
    
    xg_params = {
        "n_estimators": 250,
        "max_depth": trial.suggest_int("xg_depth", 3, 6),
        "reg_lambda": trial.suggest_float("xg_lambda", 5.0, 50.0),
        "reg_alpha": trial.suggest_float("xg_alpha", 0.1, 20.0),
        "learning_rate": trial.suggest_float("xg_lr", 0.01, 0.1),
        "objective": "count:poisson", "tree_method": "hist", "enable_categorical": True, "n_jobs": 1
    }

    lgb_params = {
        "n_estimators": 250,
        "num_leaves": trial.suggest_int("lgb_leaves", 15, 63),
        "lambda_l1": trial.suggest_float("lgb_l1", 0.0, 20.0),
        "lambda_l2": trial.suggest_float("lgb_l2", 0.0, 20.0),
        "learning_rate": trial.suggest_float("lgb_lr", 0.01, 0.1),
        "objective": "poisson", "verbose": -1, "n_jobs": 1
    }
    
    ridge_alpha = trial.suggest_float("ridge_alpha", 1.0, 100.0)

    w_cb = trial.suggest_float("ens_cb", 0.4, 1.0)
    w_xg = trial.suggest_float("ens_xg", 0.1, 0.5)
    w_lgb = trial.suggest_float("ens_lgb", 0.1, 0.5)
    w_rd = trial.suggest_float("ens_rd", 0.0, 0.2)

    all_2026_dates = sorted(DF_GLOBAL[DF_GLOBAL['year'] == 2026]['game_date'].unique())
    
    total_correct = 0
    total_valid_games = 0
    tw = w_cb + w_xg + w_lgb + w_rd
    nw = (w_cb/tw, w_xg/tw, w_lgb/tw, w_rd/tw)

    for target_date_idx, target_date in enumerate(all_2026_dates):
        train_idx = DF_GLOBAL.index[DF_GLOBAL['game_date'] < target_date]
        test_idx = DF_GLOBAL.index[DF_GLOBAL['game_date'] == target_date]
        if len(train_idx) == 0 or len(test_idx) == 0: continue
        
        sw = DF_GLOBAL.loc[train_idx, 'year'].map(weight_map)
        mu_preds = {'home': np.zeros(len(test_idx)), 'away': np.zeros(len(test_idx))}

        scaler = StandardScaler()
        X_tr_rd = np.nan_to_num(scaler.fit_transform(DF_RIDGE_ALL.loc[train_idx]))
        X_ts_rd = np.nan_to_num(scaler.transform(DF_RIDGE_ALL.loc[test_idx]))

        for target in ['homeScore', 'awayScore']:
            y_tr = DF_GLOBAL.loc[train_idx, target]
            m_cb = CatBoostRegressor(**cb_params).fit(DF_GLOBAL.loc[train_idx, FEATURES], y_tr, sample_weight=sw, cat_features=CAT_FEATURES)
            p_cb = m_cb.predict(DF_GLOBAL.loc[test_idx, FEATURES])
            m_xg = XGBRegressor(**xg_params).fit(DF_CAT_MODELS.loc[train_idx], y_tr, sample_weight=sw)
            p_xg = m_xg.predict(DF_CAT_MODELS.loc[test_idx])
            m_lgb = LGBMRegressor(**lgb_params).fit(DF_CAT_MODELS.loc[train_idx], y_tr, sample_weight=sw)
            p_lgb = m_lgb.predict(DF_CAT_MODELS.loc[test_idx])
            m_rd = Ridge(alpha=ridge_alpha).fit(X_tr_rd, y_tr, sample_weight=sw)
            p_rd = m_rd.predict(X_ts_rd)
            
            key = 'home' if target == 'homeScore' else 'away'
            mu_preds[key] = (p_cb * nw[0]) + (p_xg * nw[1]) + (p_lgb * nw[2]) + (p_rd * nw[3])
            del m_cb, m_xg, m_lgb, m_rd

        for i, row_idx in enumerate(test_idx):
            actual_row = DF_GLOBAL.loc[row_idx]
            prob = calculate_win_prob(mu_preds['home'][i], mu_preds['away'][i])
            actual = "Home" if actual_row['homeScore'] > actual_row['awayScore'] else ("Away" if actual_row['homeScore'] < actual_row['awayScore'] else "Draw")
            if actual != "Draw":
                pred = "Home" if prob >= 0.5 else "Away"
                if pred == actual: total_correct += 1
                total_valid_games += 1

        # Optuna Pruning 도입: 5일 단위로 보고하고 조기 종료 판단
        if total_valid_games > 0 and target_date_idx % 5 == 0:
            current_acc = total_correct / total_valid_games
            trial.report(current_acc, step=target_date_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

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
    table.add_row(str(current_trial), f"{last_val:.4f}" if last_val is not None else "Pruned", f"{best_val:.4f}")
    
    param_table = Table(title="[bold yellow]Best Parameters[/bold yellow]", show_header=True, header_style="bold yellow", expand=True)
    param_table.add_column("Parameter")
    param_table.add_column("Value")
    
    if len(study.trials) > 0 and study.best_trial is not None:
        for k, v in study.best_params.items():
            param_table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
            
    return Panel(table, title=f"🚀 Tuning in Progress... (Trial {current_trial})", border_style="blue"), Panel(param_table, border_style="yellow")

if __name__ == "__main__":
    N_TRIALS = 15
    console.print(Panel.fit("[bold cyan]🚀 KBO All-In Hyperparameter Optimizer (Optuna 최적화 적용)[/bold cyan]", border_style="cyan"))
    
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15)
    )
    
    with Live(console=console, refresh_per_second=4) as live:
        def callback(study, trial):
            dash1, dash2 = make_dashboard(study)
            combined = Table.grid(expand=True)
            combined.add_column()
            combined.add_column()
            combined.add_row(dash1, dash2)
            live.update(combined)

        max_workers = max(1, os.cpu_count() - 1)
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback], n_jobs=max_workers)

    console.print("\n" + "="*50)
    console.print(f"🏆 [bold green]최적화 완료 (최고 정확도: {study.best_value:.4f})[/bold green]")
    
    pd.DataFrame([study.best_params]).to_csv("best_hyperparameters.csv", index=False)
    console.print("\n📝 [bold yellow]최적 파라미터 저장 완료: best_hyperparameters.csv[/bold yellow]")
