"""경기 전 불변 예측 로그의 전향적 확률 성능을 집계한다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline_config import EVALUATIONS_DIR, OPERATIONS_DIR, RAW_DATA_DIR


def prepare_evaluation_rows(
    prediction_log: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """경기 시작 전에 최초 저장된 비무승부 예측만 평가 대상으로 만든다."""

    required = {
        "record_type",
        "s_no",
        "recorded_at",
        "game_datetime",
        "home_win_probability",
        "model_type",
    }
    missing = required.difference(prediction_log.columns)
    if missing:
        raise ValueError(f"예측 로그 필수 열 누락: {sorted(missing)}")

    log = prediction_log.loc[
        prediction_log["record_type"].eq("prediction")
    ].copy()
    log["recorded_at"] = pd.to_datetime(
        log["recorded_at"], errors="coerce", utc=True
    )
    log["game_datetime"] = pd.to_datetime(
        log["game_datetime"], errors="coerce", utc=True
    )
    log = log.loc[
        log["recorded_at"].notna()
        & log["game_datetime"].notna()
        & log["recorded_at"].lt(log["game_datetime"])
    ]
    log = log.sort_values("recorded_at").drop_duplicates("s_no", keep="first")
    result = log.merge(
        games[["s_no", "homeScore", "awayScore"]],
        on="s_no",
        how="inner",
    )
    home_score = pd.to_numeric(result["homeScore"], errors="coerce")
    away_score = pd.to_numeric(result["awayScore"], errors="coerce")
    result = result.loc[
        home_score.notna()
        & away_score.notna()
        & home_score.ne(away_score)
    ].copy()
    result["target_home_win"] = (
        home_score.loc[result.index].gt(away_score.loc[result.index]).astype(int)
    )
    return result.reset_index(drop=True)


def evaluate_prediction_log(rows: pd.DataFrame) -> pd.DataFrame:
    """주간·월간·모델 유형별 확률 품질과 대체 모델 사용률을 계산한다."""

    if rows.empty:
        return pd.DataFrame()
    frame = rows.copy()
    seoul_time = frame["game_datetime"].dt.tz_convert("Asia/Seoul")
    frame["week"] = seoul_time.dt.to_period("W").astype(str)
    frame["month"] = seoul_time.dt.to_period("M").astype(str)
    probability = pd.to_numeric(
        frame["home_win_probability"], errors="coerce"
    ).clip(1e-6, 1 - 1e-6)
    target = frame["target_home_win"].astype(float)
    frame["_log_loss"] = -(
        target * np.log(probability)
        + (1 - target) * np.log(1 - probability)
    )
    frame["_brier"] = (probability - target) ** 2
    frame["_correct"] = probability.ge(0.5).eq(target.eq(1)).astype(float)
    frame["_fallback"] = frame["model_type"].eq(
        "fallback_recent10"
    ).astype(float)

    reports = []
    for period_type in ("week", "month"):
        grouped = (
            frame.groupby([period_type, "model_type"], dropna=False)
            .agg(
                n_games=("s_no", "size"),
                log_loss=("_log_loss", "mean"),
                brier_score=("_brier", "mean"),
                accuracy=("_correct", "mean"),
                fallback_rate=("_fallback", "mean"),
            )
            .reset_index()
            .rename(columns={period_type: "period"})
        )
        grouped["period_type"] = period_type
        reports.append(grouped)
    return pd.concat(reports, ignore_index=True)


def main() -> None:
    log_path = OPERATIONS_DIR / "prediction_log.csv"
    if not log_path.exists():
        print("prediction_log.csv가 없어 평가를 건너뜁니다.")
        return
    log = pd.read_csv(log_path)
    games = pd.read_csv(RAW_DATA_DIR / "games_master.csv")
    rows = prepare_evaluation_rows(log, games)
    report = evaluate_prediction_log(rows)
    report.to_csv(
        EVALUATIONS_DIR / "prediction_performance_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
