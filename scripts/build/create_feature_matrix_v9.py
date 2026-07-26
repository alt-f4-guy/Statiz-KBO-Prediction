"""현재 로컬 원천 데이터로 누수 없는 v9 학습 데이터셋을 생성한다."""

from __future__ import annotations

import pandas as pd

from feature_matrix_v9 import build_feature_matrix_v9
from pipeline_config import (
    FINAL_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REFERENCE_DATA_DIR,
)


def main() -> None:
    games = pd.read_csv(RAW_DATA_DIR / "games_master.csv")
    lineups = pd.read_csv(RAW_DATA_DIR / "lineups.csv")
    rosters = pd.read_csv(RAW_DATA_DIR / "rosters.csv")
    day = pd.read_csv(PROCESSED_DATA_DIR / "player_day_processed_v2.csv")
    season = pd.read_csv(PROCESSED_DATA_DIR / "player_season_processed_v2.csv")

    data, coverage, constants = build_feature_matrix_v9(
        games, lineups, rosters, day, season
    )
    FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(
        FINAL_DATA_DIR / "final_training_set_v9.csv",
        index=False,
        encoding="utf-8-sig",
    )
    coverage.to_csv(
        FINAL_DATA_DIR / "feature_coverage_v9.csv",
        index=False,
        encoding="utf-8-sig",
    )
    constants.to_csv(
        REFERENCE_DATA_DIR / "kbo_year_constants.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"v9 학습 데이터: {len(data):,}경기")
    print(f"평균 피처 결측률: {coverage['feature_missing_rate'].mean():.2%}")


if __name__ == "__main__":
    main()
