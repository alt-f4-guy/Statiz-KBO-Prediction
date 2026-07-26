"""고정된 운영 모델의 2026년 비무승부 경기 확률을 재현한다."""

from __future__ import annotations

import joblib
import pandas as pd

from classifier_model import prepare_classifier_frame, probability_metrics
from pipeline_config import EVALUATIONS_DIR, FINAL_DATA_DIR, MODEL_DIR
from time_splits import make_temporal_split_manifest


def run_backtest() -> tuple[pd.DataFrame, dict[str, float]]:
    data = pd.read_csv(FINAL_DATA_DIR / "final_training_set_v9.csv")
    manifest = make_temporal_split_manifest(data)
    frame = prepare_classifier_frame(data)
    final_test = frame.loc[
        frame["s_no"].isin(manifest["final_test_s_nos"])
    ].copy()
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    probability = model.predict_proba(final_test[model.feature_columns])[:, 1]
    result = pd.DataFrame(
        {
            "s_no": final_test["s_no"].astype(int).to_numpy(),
            "game_datetime": final_test["game_datetime"].to_numpy(),
            "home_win_probability": probability,
            "predicted_home_win": probability >= 0.5,
            "actual_home_win": final_test["target_home_win"].to_numpy(),
        }
    )
    result["correct"] = result["predicted_home_win"].astype(int).eq(
        result["actual_home_win"]
    )
    metrics = probability_metrics(
        result["actual_home_win"],
        result["home_win_probability"].to_numpy(),
    )
    output_path = EVALUATIONS_DIR / "backtest_results_v9.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )
    return result, metrics


def main() -> None:
    _, metrics = run_backtest()
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
