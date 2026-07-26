"""직접 분류기와 득점 분포 모델을 동일 경기·확률 지표로 비교한다."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from classifier_model import prepare_classifier_frame, probability_metrics
from pipeline_config import EVALUATIONS_DIR, FINAL_DATA_DIR, MODEL_DIR


class ModelComparisonError(ValueError):
    """모델 간 공정 비교 계약 위반."""


def assert_same_evaluation_games(predictions: pd.DataFrame) -> None:
    """모든 모델이 정확히 같은 경기 ID를 평가했는지 확인한다."""

    groups = {
        model: set(group["s_no"].astype(int))
        for model, group in predictions.groupby("model")
    }
    if not groups:
        raise ModelComparisonError("비교할 예측이 없습니다.")
    reference_model, reference_ids = next(iter(groups.items()))
    for model, ids in groups.items():
        if ids != reference_ids:
            raise ModelComparisonError(
                f"평가 경기 ID가 동일하지 않습니다: {reference_model}, {model}"
            )


def select_operating_model(summary: pd.DataFrame) -> str:
    """개발 구간 지표만으로 운영 모델을 선택한다."""

    candidates = (
        summary.assign(
            development_calibration_distance=(
                summary["development_calibration_intercept"].abs()
                + (summary["development_calibration_slope"] - 1).abs()
            )
        )
        .sort_values(
            [
                "development_log_loss",
                "development_brier_score",
                "development_calibration_distance",
                "model",
            ]
        )
        .reset_index(drop=True)
    )
    return str(candidates.loc[0, "model"])


def _model_summary(results: pd.DataFrame) -> pd.DataFrame:
    development = (
        results.loc[results["split"].str.startswith("development")]
        .groupby("model", as_index=False)[
            ["log_loss", "brier_score", "calibration_intercept", "calibration_slope"]
        ]
        .mean()
        .rename(
            columns={
                "log_loss": "development_log_loss",
                "brier_score": "development_brier_score",
                "calibration_intercept": "development_calibration_intercept",
                "calibration_slope": "development_calibration_slope",
            }
        )
    )
    final = results.loc[results["split"].eq("final_2026")].rename(
        columns={
            "log_loss": "final_log_loss",
            "brier_score": "final_brier_score",
            "calibration_intercept": "final_calibration_intercept",
            "calibration_slope": "final_calibration_slope",
        }
    )
    keep = [
        "model",
        "final_log_loss",
        "final_brier_score",
        "final_calibration_intercept",
        "final_calibration_slope",
        "roc_auc",
        "accuracy",
    ]
    for optional in ("score_mae", "score_rmse"):
        if optional in final.columns:
            keep.append(optional)
    return development.merge(final[keep], on="model", how="inner")


def _calibration_bins(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["probability_bin"] = pd.cut(
        frame["home_win_probability"],
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
    )
    return (
        frame.groupby(["model", "probability_bin"], observed=True)
        .agg(
            n_games=("s_no", "size"),
            mean_probability=("home_win_probability", "mean"),
            actual_home_win_rate=("target_home_win", "mean"),
        )
        .reset_index()
    )


def run_comparison() -> tuple[pd.DataFrame, dict]:
    classifier_results = pd.read_csv(
        EVALUATIONS_DIR / "classifier_comparison_results.csv"
    )
    score_results = pd.read_csv(
        EVALUATIONS_DIR / "score_model_comparison_results.csv"
    )
    classifier_summary = _model_summary(classifier_results)
    classifier_summary["family"] = "direct_classifier"
    score_summary = _model_summary(score_results)
    score_summary["family"] = "score_distribution"
    summary = pd.concat(
        [classifier_summary, score_summary],
        ignore_index=True,
        sort=False,
    )

    classifier_predictions = pd.read_csv(
        FINAL_DATA_DIR / "classifier_predictions_2026.csv"
    )
    score_predictions = pd.read_csv(
        FINAL_DATA_DIR / "score_model_predictions_2026.csv"
    )
    predictions = pd.concat(
        [
            classifier_predictions[
                [
                    "s_no",
                    "model",
                    "home_win_probability",
                    "target_home_win",
                ]
            ],
            score_predictions[
                [
                    "s_no",
                    "model",
                    "home_win_probability",
                    "target_home_win",
                ]
            ],
        ],
        ignore_index=True,
    )
    assert_same_evaluation_games(predictions)

    training = prepare_classifier_frame(
        pd.read_csv(FINAL_DATA_DIR / "final_training_set_v9.csv")
    )
    baseline_probability = training.loc[
        training["year"].lt(2026), "target_home_win"
    ].mean()
    target = (
        predictions.loc[predictions["model"].eq(predictions["model"].iloc[0])]
        .sort_values("s_no")["target_home_win"]
        .to_numpy()
    )
    baseline_metrics = probability_metrics(
        target, np.full(len(target), baseline_probability)
    )
    baseline_row = {
        "model": "constant_home_rate",
        "family": "baseline",
        "development_log_loss": np.nan,
        "development_brier_score": np.nan,
        "final_log_loss": baseline_metrics["log_loss"],
        "final_brier_score": baseline_metrics["brier_score"],
        "final_calibration_intercept": baseline_metrics[
            "calibration_intercept"
        ],
        "final_calibration_slope": baseline_metrics["calibration_slope"],
        "roc_auc": baseline_metrics["roc_auc"],
        "accuracy": baseline_metrics["accuracy"],
    }
    summary = pd.concat([summary, pd.DataFrame([baseline_row])], ignore_index=True)
    selected = select_operating_model(
        summary.loc[summary["family"].ne("baseline")]
    )

    summary.to_csv(
        EVALUATIONS_DIR / "model_comparison_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _calibration_bins(predictions).to_csv(
        EVALUATIONS_DIR / "model_calibration_bins.csv",
        index=False,
        encoding="utf-8-sig",
    )

    family = summary.set_index("model").loc[selected, "family"]
    source_model = (
        MODEL_DIR / "best_classifier.joblib"
        if family == "direct_classifier"
        else MODEL_DIR / "best_score_model.joblib"
    )
    target_model = MODEL_DIR / "best_model.joblib"
    shutil.copyfile(source_model, target_model)
    metadata = {
        "selected_model": selected,
        "model_family": family,
        "selection_rule": (
            "개발 폴드의 로그 손실, 브라이어 점수, 보정 거리 순서; "
            "2026 최종 테스트 지표는 선택에 사용하지 않음"
        ),
        "random_state": 42,
        "data_version": "final_training_set_v9",
        "split_manifest": str(
            FINAL_DATA_DIR / "time_split_manifest.json"
        ),
    }
    (MODEL_DIR / "best_model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary, metadata


def main() -> None:
    summary, metadata = run_comparison()
    columns = [
        "model",
        "family",
        "development_log_loss",
        "development_brier_score",
        "final_log_loss",
        "final_brier_score",
        "final_calibration_intercept",
        "final_calibration_slope",
        "roc_auc",
        "accuracy",
    ]
    print(summary[columns].to_string(index=False))
    print(f"운영 후보: {metadata['selected_model']}")


if __name__ == "__main__":
    main()
