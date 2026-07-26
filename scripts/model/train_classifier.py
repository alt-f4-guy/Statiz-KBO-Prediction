"""공통 순차 분할로 직접 승패 분류기를 비교·보정·저장한다."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from classifier_model import (
    CalibratedBinaryModel,
    SigmoidCalibrator,
    build_classifier,
    prepare_classifier_frame,
    probability_metrics,
    select_model_features,
)
from pipeline_config import EVALUATIONS_DIR, FINAL_DATA_DIR, MODEL_DIR, TUNING_DIR
from time_splits import (
    make_temporal_split_manifest,
    save_split_manifest,
    split_calibration_tail,
)


SPLIT_PATH = FINAL_DATA_DIR / "time_split_manifest.json"
TUNED_PARAMS_PATH = TUNING_DIR / "best_classifier_hyperparameters.csv"


def _rows_for_ids(frame: pd.DataFrame, ids: list[int]) -> pd.DataFrame:
    return frame.loc[frame["s_no"].isin(ids)].copy()


def _fit_calibrated_model(
    kind: str,
    features: list[str],
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    params: dict | None = None,
) -> CalibratedBinaryModel:
    model = build_classifier(kind, features, params)
    model.fit(train[features], train["target_home_win"])
    raw_probability = model.predict_proba(calibration[features])[:, 1]
    calibrator = SigmoidCalibrator().fit(
        raw_probability, calibration["target_home_win"]
    )
    return CalibratedBinaryModel(model, calibrator, features)


def _evaluate_classifier_fold(
    kind: str,
    features: list[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    params: dict | None = None,
) -> pd.Series:
    base_ids, calibrator_ids = split_calibration_tail(train)
    base_train = _rows_for_ids(train, base_ids)
    fold_calibration = _rows_for_ids(train, calibrator_ids)

    if fold_calibration["target_home_win"].nunique() < 2:
        raise ValueError("보정 구간에 두 클래스가 모두 존재해야 합니다.")
    if validation["target_home_win"].nunique() < 2:
        raise ValueError("검증 구간에 두 클래스가 모두 존재해야 합니다.")

    calibrated_model = _fit_calibrated_model(
        kind, features, base_train, fold_calibration, params
    )
    return pd.Series(
        calibrated_model.predict_proba(validation[features])[:, 1],
        index=validation.index,
    )



def run_training() -> tuple[pd.DataFrame, dict]:
    data = pd.read_csv(FINAL_DATA_DIR / "final_training_set_v9.csv")
    manifest = make_temporal_split_manifest(data)
    save_split_manifest(manifest, SPLIT_PATH)
    frame = prepare_classifier_frame(data)
    features = select_model_features(frame)
    catboost_params = None
    if TUNED_PARAMS_PATH.exists():
        tuned = pd.read_csv(TUNED_PARAMS_PATH).iloc[0].to_dict()
        tuned.pop("development_log_loss", None)
        for integer_column in ("iterations", "depth"):
            if integer_column in tuned:
                tuned[integer_column] = int(tuned[integer_column])
        catboost_params = tuned

    results: list[dict] = []
    model_kinds = ["logistic", "catboost"]
    for kind in model_kinds:
        for fold_index, fold in enumerate(manifest["development_folds"], start=1):
            train = _rows_for_ids(frame, fold["train_s_nos"])
            validation = _rows_for_ids(frame, fold["validation_s_nos"])
            probability = _evaluate_classifier_fold(
                kind,
                features,
                train,
                validation,
                catboost_params if kind == "catboost" else None,
            )
            metrics = probability_metrics(
                validation["target_home_win"], probability
            )
            results.append(
                {
                    "model": kind,
                    "split": f"development_fold_{fold_index}",
                    "n_games": len(validation),
                    "home_win_rate": validation["target_home_win"].mean(),
                    **metrics,
                }
            )

    result_frame = pd.DataFrame(results)
    development_summary = (
        result_frame.groupby("model", as_index=False)[
            ["log_loss", "brier_score"]
        ]
        .mean()
        .sort_values(["log_loss", "brier_score"])
    )
    selected_kind = development_summary.iloc[0]["model"]

    train_ids = (
        manifest["initial_train_s_nos"] + manifest["development_s_nos"]
    )
    train = _rows_for_ids(frame, train_ids)
    calibration = _rows_for_ids(frame, manifest["calibration_s_nos"])
    final_test = _rows_for_ids(frame, manifest["final_test_s_nos"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final_predictions = []
    fitted_models: dict[str, CalibratedBinaryModel] = {}
    for kind in model_kinds:
        calibrated_model = _fit_calibrated_model(
            kind,
            features,
            train,
            calibration,
            catboost_params if kind == "catboost" else None,
        )
        fitted_models[kind] = calibrated_model
        probability = calibrated_model.predict_proba(final_test[features])[:, 1]
        metrics = probability_metrics(
            final_test["target_home_win"], probability
        )
        results.append(
            {
                "model": kind,
                "split": "final_2026",
                "n_games": len(final_test),
                "home_win_rate": final_test["target_home_win"].mean(),
                **metrics,
            }
        )
        final_predictions.append(
            pd.DataFrame(
                {
                    "s_no": final_test["s_no"].to_numpy(),
                    "model": kind,
                    "home_win_probability": probability,
                    "target_home_win": final_test[
                        "target_home_win"
                    ].to_numpy(),
                }
            )
        )

    selected_model = fitted_models[selected_kind]
    joblib.dump(selected_model, MODEL_DIR / "best_classifier.joblib")
    pd.concat(final_predictions, ignore_index=True).to_csv(
        FINAL_DATA_DIR / "classifier_predictions_2026.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(
        EVALUATIONS_DIR / "classifier_comparison_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "model_family": "direct_classifier",
        "selected_classifier": selected_kind,
        "selection_metric": "mean development log_loss",
        "random_state": 42,
        "training_s_nos": train["s_no"].astype(int).tolist(),
        "calibration_s_nos": calibration["s_no"].astype(int).tolist(),
        "final_test_s_nos": final_test["s_no"].astype(int).tolist(),
        "feature_columns": features,
        "catboost_parameters": catboost_params,
        "class_distribution": {
            "train_home_win_rate": float(train["target_home_win"].mean()),
            "calibration_home_win_rate": float(
                calibration["target_home_win"].mean()
            ),
            "final_home_win_rate": float(final_test["target_home_win"].mean()),
        },
    }
    (MODEL_DIR / "classifier_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result_frame, metadata


def main() -> None:
    results, metadata = run_training()
    summary = results.loc[results["split"].eq("final_2026"), [
        "model",
        "log_loss",
        "brier_score",
        "roc_auc",
        "accuracy",
    ]]
    print(summary.to_string(index=False))
    print(f"개발 구간 선택 분류기: {metadata['selected_classifier']}")


if __name__ == "__main__":
    main()
