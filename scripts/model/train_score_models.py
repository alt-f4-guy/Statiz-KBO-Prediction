"""득점 분포 모델을 공통 순차 분할에서 학습·보정·평가한다."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from classifier_model import (
    SigmoidCalibrator,
    prepare_classifier_frame,
    probability_metrics,
    select_model_features,
)
from pipeline_config import EVALUATIONS_DIR, FINAL_DATA_DIR, MODEL_DIR
from score_models import (
    build_score_model,
    conditional_skellam_home_probability,
    score_prediction_metrics,
)
from time_splits import make_temporal_split_manifest


def _rows_for_ids(frame: pd.DataFrame, ids: list[int]) -> pd.DataFrame:
    return frame.loc[frame["s_no"].isin(ids)].copy()


def _probability_rows(
    frame: pd.DataFrame,
    home_mean: np.ndarray,
    away_mean: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    non_draw = frame["homeScore"].ne(frame["awayScore"]).to_numpy()
    probability = conditional_skellam_home_probability(
        home_mean[non_draw], away_mean[non_draw]
    )
    prepared = prepare_classifier_frame(frame)
    return prepared, probability


def run_training() -> pd.DataFrame:
    data = pd.read_csv(FINAL_DATA_DIR / "final_training_set_v9.csv")
    manifest = make_temporal_split_manifest(data)
    features = select_model_features(data)
    model_kinds = ["poisson_catboost", "negative_binomial"]
    results: list[dict] = []

    for kind in model_kinds:
        for fold_index, fold in enumerate(manifest["development_folds"], start=1):
            train = _rows_for_ids(data, fold["train_s_nos"])
            validation = _rows_for_ids(data, fold["validation_s_nos"])
            model = build_score_model(kind, features)
            model.fit(
                train[features], train["homeScore"], train["awayScore"]
            )
            home_mean, away_mean = model.predict(validation[features])
            probability_frame, probability = _probability_rows(
                validation, home_mean, away_mean
            )
            results.append(
                {
                    "model": kind,
                    "split": f"development_fold_{fold_index}",
                    "n_games": len(probability_frame),
                    **probability_metrics(
                        probability_frame["target_home_win"], probability
                    ),
                    **score_prediction_metrics(
                        validation["homeScore"],
                        validation["awayScore"],
                        home_mean,
                        away_mean,
                    ),
                }
            )

    summary = (
        pd.DataFrame(results)
        .groupby("model", as_index=False)[["log_loss", "brier_score"]]
        .mean()
        .sort_values(["log_loss", "brier_score"])
    )
    selected_kind = summary.iloc[0]["model"]
    train_ids = (
        manifest["initial_train_s_nos"] + manifest["development_s_nos"]
    )
    train = _rows_for_ids(data, train_ids)
    calibration = _rows_for_ids(data, manifest["calibration_s_nos"])
    final_test = _rows_for_ids(data, manifest["final_test_s_nos"])
    final_predictions = []
    fitted_models = {}

    for kind in model_kinds:
        model = build_score_model(kind, features)
        model.fit(train[features], train["homeScore"], train["awayScore"])
        calibration_home, calibration_away = model.predict(
            calibration[features]
        )
        calibration_frame, raw_calibration_probability = _probability_rows(
            calibration, calibration_home, calibration_away
        )
        calibrator = SigmoidCalibrator().fit(
            raw_calibration_probability,
            calibration_frame["target_home_win"],
        )
        final_home, final_away = model.predict(final_test[features])
        final_probability_frame, raw_final_probability = _probability_rows(
            final_test, final_home, final_away
        )
        final_probability = calibrator.predict(raw_final_probability)
        score_metrics = score_prediction_metrics(
            final_test["homeScore"],
            final_test["awayScore"],
            final_home,
            final_away,
        )
        results.append(
            {
                "model": kind,
                "split": "final_2026",
                "n_games": len(final_probability_frame),
                **probability_metrics(
                    final_probability_frame["target_home_win"],
                    final_probability,
                ),
                **score_metrics,
            }
        )
        fitted_models[kind] = {
            "score_model": model,
            "calibrator": calibrator,
            "feature_columns": features,
        }
        non_draw = final_test["homeScore"].ne(final_test["awayScore"])
        final_predictions.append(
            pd.DataFrame(
                {
                    "s_no": final_test.loc[non_draw, "s_no"].to_numpy(),
                    "model": kind,
                    "home_score_mean": final_home[non_draw.to_numpy()],
                    "away_score_mean": final_away[non_draw.to_numpy()],
                    "home_win_probability": final_probability,
                    "target_home_win": final_probability_frame[
                        "target_home_win"
                    ].to_numpy(),
                }
            )
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        fitted_models[selected_kind],
        MODEL_DIR / "best_score_model.joblib",
    )
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(
        EVALUATIONS_DIR / "score_model_comparison_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(final_predictions, ignore_index=True).to_csv(
        FINAL_DATA_DIR / "score_model_predictions_2026.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result_frame


def main() -> None:
    results = run_training()
    columns = [
        "model",
        "log_loss",
        "brier_score",
        "score_mae",
        "score_rmse",
        "accuracy",
    ]
    print(
        results.loc[results["split"].eq("final_2026"), columns].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
