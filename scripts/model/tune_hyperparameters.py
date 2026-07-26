"""2025년 순차 검증 로그 손실로 CatBoost 분류기를 튜닝한다."""

from __future__ import annotations

import os

import optuna
import pandas as pd
from sklearn.metrics import log_loss

from classifier_model import (
    build_classifier,
    prepare_classifier_frame,
    select_model_features,
)
from pipeline_config import FINAL_DATA_DIR, TUNING_DIR
from time_splits import make_temporal_split_manifest


OUTPUT_PATH = TUNING_DIR / "best_classifier_hyperparameters.csv"


def build_tuning_folds(
    data: pd.DataFrame,
) -> tuple[list[tuple[pd.DataFrame, pd.DataFrame]], list[str]]:
    """2026년과 보정 구간을 제외한 순차 개발 폴드를 반환한다."""

    manifest = make_temporal_split_manifest(data)
    frame = prepare_classifier_frame(data)
    features = select_model_features(frame)
    folds = []
    for fold in manifest["development_folds"]:
        train = frame.loc[frame["s_no"].isin(fold["train_s_nos"])].copy()
        validation = frame.loc[
            frame["s_no"].isin(fold["validation_s_nos"])
        ].copy()
        if train["year"].eq(2026).any() or validation["year"].eq(2026).any():
            raise ValueError("튜닝 폴드에 2026년 데이터가 포함됐습니다.")
        folds.append((train, validation))
    return folds, features


def _objective_factory(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    features: list[str],
):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 250, 700, step=50),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.10, log=True
            ),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", 1.0, 12.0, log=True
            ),
            "random_strength": trial.suggest_float(
                "random_strength", 0.0, 2.0
            ),
        }
        losses = []
        for train, validation in folds:
            model = build_classifier("catboost", features, params)
            model.fit(train[features], train["target_home_win"])
            probability = model.predict_proba(validation[features])[:, 1]
            losses.append(
                log_loss(
                    validation["target_home_win"],
                    probability,
                    labels=[0, 1],
                )
            )
        return float(sum(losses) / len(losses))

    return objective


def run_tuning(n_trials: int = 30) -> dict:
    data = pd.read_csv(FINAL_DATA_DIR / "final_training_set_v9.csv")
    folds, features = build_tuning_folds(data)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        _objective_factory(folds, features),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    output = {**study.best_params, "development_log_loss": study.best_value}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([output]).to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    return output


def main() -> None:
    n_trials = int(os.getenv("KBO_OPTUNA_TRIALS", "30"))
    best = run_tuning(n_trials)
    print(pd.DataFrame([best]).to_string(index=False))


if __name__ == "__main__":
    main()
