"""2023~2025년 대체 모델의 순차 확률 성능을 별도로 기록한다."""

from __future__ import annotations

import pandas as pd

from classifier_model import probability_metrics
from fallback_recent10 import backtest_recent_ten
from game_time import build_game_datetime_reference
from pipeline_config import EVALUATIONS_DIR, RAW_DATA_DIR


def main() -> None:
    games = pd.read_csv(RAW_DATA_DIR / "games_master.csv")
    games = games.loc[
        pd.to_numeric(games["year"], errors="coerce").between(2023, 2025)
    ].copy()
    reference = build_game_datetime_reference(games)
    games = games.merge(
        reference[
            ["s_no", "game_datetime", "feature_cutoff_datetime"]
        ],
        on="s_no",
        how="inner",
    )
    result = backtest_recent_ten(games)
    metrics = probability_metrics(
        result["target_home_win"], result["home_win_probability"].to_numpy()
    )
    result.to_csv(
        EVALUATIONS_DIR / "fallback_recent10_backtest_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([{"model": "fallback_recent10", **metrics}]).to_csv(
        EVALUATIONS_DIR / "fallback_recent10_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
