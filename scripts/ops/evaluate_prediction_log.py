"""고정된 전향적 배포의 경기 전 예측과 운영 품질을 평가한다."""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
import pandas as pd

from classifier_model import probability_metrics
from pipeline_config import EVALUATIONS_DIR, MODEL_DIR, OPERATIONS_DIR, RAW_DATA_DIR


AUDIT_COLUMNS = {
    "record_type",
    "s_no",
    "recorded_at",
    "game_datetime",
    "home_win_probability",
    "model_type",
    "deployment_id",
    "evaluation_role",
    "prospective_start_date",
    "lineup_complete",
    "feature_prior_usage_rate",
    "api_status",
}


def prepare_evaluation_rows(
    prediction_log: pd.DataFrame,
    games: pd.DataFrame,
    *,
    deployment_id: str,
) -> pd.DataFrame:
    """지정 배포의 경기 전 최초 전향적 비무승부 예측만 선택한다."""

    missing = AUDIT_COLUMNS.difference(prediction_log.columns)
    if missing:
        raise ValueError(f"예측 로그 필수 감사 열 누락: {sorted(missing)}")

    log = prediction_log.copy()
    log["recorded_at"] = pd.to_datetime(
        log["recorded_at"], errors="coerce", utc=True
    )
    log["game_datetime"] = pd.to_datetime(
        log["game_datetime"], errors="coerce", utc=True
    )
    start_dates = pd.to_datetime(
        log["prospective_start_date"], errors="coerce"
    ).dt.date
    recorded_dates = (
        log["recorded_at"].dt.tz_convert("Asia/Seoul").dt.date
    )

    predictions = log.loc[
        log["record_type"].eq("prediction")
        & log["deployment_id"].eq(deployment_id)
        & log["evaluation_role"].eq("prospective_holdout")
        & log["recorded_at"].notna()
        & log["game_datetime"].notna()
        & log["recorded_at"].lt(log["game_datetime"])
        & start_dates.notna()
        & recorded_dates.ge(start_dates)
    ].copy()
    predictions = (
        predictions.sort_values("recorded_at")
        .drop_duplicates("s_no", keep="first")
    )

    deliveries = log.loc[
        log["record_type"].eq("delivery")
        & log["deployment_id"].eq(deployment_id),
        ["s_no", "api_status"],
    ].copy()
    deliveries["delivery_success"] = deliveries["api_status"].eq(
        "success"
    ).astype(float)
    delivery_status = (
        deliveries.groupby("s_no", as_index=False)["delivery_success"].max()
        if not deliveries.empty
        else pd.DataFrame(columns=["s_no", "delivery_success"])
    )
    predictions = predictions.merge(delivery_status, on="s_no", how="left")
    predictions["delivery_success"] = predictions["delivery_success"].fillna(0.0)

    result = predictions.merge(
        games[["s_no", "homeScore", "awayScore"]],
        on="s_no",
        how="inner",
    )
    home_score = pd.to_numeric(result["homeScore"], errors="coerce")
    away_score = pd.to_numeric(result["awayScore"], errors="coerce")
    valid = (
        home_score.notna()
        & away_score.notna()
        & home_score.ne(away_score)
    )
    result = result.loc[valid].copy()
    result["target_home_win"] = (
        home_score.loc[valid].gt(away_score.loc[valid]).astype(int)
    )
    return result.reset_index(drop=True)


def _safe_probability_metrics(rows: pd.DataFrame) -> dict[str, float]:
    target = pd.to_numeric(rows["target_home_win"], errors="raise").astype(int)
    probability = pd.to_numeric(
        rows["home_win_probability"], errors="raise"
    ).clip(1e-6, 1 - 1e-6)
    log_loss_value = -(
        target * np.log(probability)
        + (1 - target) * np.log(1 - probability)
    ).mean()
    metrics = {
        "log_loss": float(log_loss_value),
        "brier_score": float(((probability - target) ** 2).mean()),
        "accuracy": float(probability.ge(0.5).eq(target.eq(1)).mean()),
        "calibration_intercept": float("nan"),
        "calibration_slope": float("nan"),
        "roc_auc": float("nan"),
    }
    if len(rows) >= 10 and target.nunique() == 2:
        full_metrics = probability_metrics(target, probability.to_numpy())
        metrics.update(full_metrics)
    return metrics


def paired_day_block_bootstrap(
    rows: pd.DataFrame,
    baseline_probability: float,
    *,
    iterations: int = 2_000,
    seed: int = 42,
) -> dict[str, float]:
    """경기일 블록을 짝지어 재표본하고 모델-기준선 손실 차이를 계산한다."""

    if rows.empty:
        return {
            "mean_log_loss_delta": float("nan"),
            "log_loss_delta_ci_low": float("nan"),
            "log_loss_delta_ci_high": float("nan"),
            "mean_brier_delta": float("nan"),
            "brier_delta_ci_low": float("nan"),
            "brier_delta_ci_high": float("nan"),
        }
    baseline = float(np.clip(baseline_probability, 1e-6, 1 - 1e-6))
    target = pd.to_numeric(rows["target_home_win"], errors="raise").to_numpy()
    probability = np.clip(
        pd.to_numeric(
            rows["home_win_probability"], errors="raise"
        ).to_numpy(dtype=float),
        1e-6,
        1 - 1e-6,
    )
    baseline_array = np.full(len(rows), baseline)
    model_log_loss = -(
        target * np.log(probability)
        + (1 - target) * np.log(1 - probability)
    )
    baseline_log_loss = -(
        target * np.log(baseline_array)
        + (1 - target) * np.log(1 - baseline_array)
    )
    frame = pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                rows["game_datetime"], errors="raise", utc=True
            ).dt.tz_convert("Asia/Seoul").dt.date,
            "log_delta": model_log_loss - baseline_log_loss,
            "brier_delta": (
                (probability - target) ** 2
                - (baseline_array - target) ** 2
            ),
        }
    )
    blocks = [
        group[["log_delta", "brier_delta"]].to_numpy()
        for _, group in frame.groupby("game_date", sort=True)
    ]
    rng = np.random.default_rng(seed)
    log_samples = np.empty(iterations, dtype=float)
    brier_samples = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled_indices = rng.integers(0, len(blocks), size=len(blocks))
        sampled = np.concatenate(
            [blocks[block_index] for block_index in sampled_indices],
            axis=0,
        )
        log_samples[index] = sampled[:, 0].mean()
        brier_samples[index] = sampled[:, 1].mean()

    return {
        "mean_log_loss_delta": float(frame["log_delta"].mean()),
        "log_loss_delta_ci_low": float(np.quantile(log_samples, 0.025)),
        "log_loss_delta_ci_high": float(np.quantile(log_samples, 0.975)),
        "mean_brier_delta": float(frame["brier_delta"].mean()),
        "brier_delta_ci_low": float(np.quantile(brier_samples, 0.025)),
        "brier_delta_ci_high": float(np.quantile(brier_samples, 0.975)),
    }


def operating_decision(metrics: Mapping[str, float]) -> str:
    """서로 겹칠 수 있는 운영 조건을 고정 우선순위로 판정한다."""

    if int(metrics["n_games"]) < 100:
        return "추가 관찰"
    if (
        float(metrics["monthly_fallback_rate"]) > 0.20
        or float(metrics["prior_usage_increase"]) > 0.10
    ):
        return "데이터 파이프라인 점검"
    if int(metrics["n_games"]) < 300:
        return "추가 관찰"
    if (
        float(metrics["log_loss_delta_ci_low"]) > 0
        and float(metrics["brier_delta_ci_low"]) > 0
    ):
        return "모델 재검토"

    intercept = float(metrics["calibration_intercept"])
    slope = float(metrics["calibration_slope"])
    if (
        np.isfinite(intercept)
        and np.isfinite(slope)
        and (abs(intercept) > 0.10 or not 0.8 <= slope <= 1.2)
    ):
        return "재보정 검토"

    log_low = float(metrics["log_loss_delta_ci_low"])
    log_high = float(metrics["log_loss_delta_ci_high"])
    brier_low = float(metrics["brier_delta_ci_low"])
    brier_high = float(metrics["brier_delta_ci_high"])
    one_better = log_high < 0 or brier_high < 0
    neither_clearly_worse = log_low <= 0 and brier_low <= 0
    if not one_better or not neither_clearly_worse:
        return "추가 관찰"
    return "유지"


def _group_metrics(
    rows: pd.DataFrame,
    *,
    baseline_probability: float,
) -> dict[str, float]:
    metrics = _safe_probability_metrics(rows)
    metrics.update(
        paired_day_block_bootstrap(rows, baseline_probability)
    )
    metrics.update(
        {
            "n_games": int(len(rows)),
            "fallback_rate": float(
                rows["model_type"].eq("fallback_recent10").mean()
            ),
            "api_success_rate": float(rows["delivery_success"].mean()),
            "lineup_completion_rate": float(
                pd.to_numeric(
                    rows["lineup_complete"], errors="coerce"
                ).mean()
            ),
            "feature_prior_usage_rate": float(
                pd.to_numeric(
                    rows["feature_prior_usage_rate"], errors="coerce"
                ).mean()
            ),
            "baseline_home_probability": float(baseline_probability),
        }
    )
    return metrics


def evaluate_prediction_log(
    rows: pd.DataFrame,
    baseline_probability: float,
) -> pd.DataFrame:
    """전체와 모델 유형별 주간·월간·전체 운영 및 확률 지표를 계산한다."""

    if rows.empty:
        return pd.DataFrame()
    frame = rows.copy()
    game_time = pd.to_datetime(
        frame["game_datetime"], errors="raise", utc=True
    )
    seoul_time = game_time.dt.tz_convert("Asia/Seoul")
    local_time = seoul_time.dt.tz_localize(None)
    frame["week"] = local_time.dt.to_period("W").astype(str)
    frame["month"] = local_time.dt.to_period("M").astype(str)

    reports: list[dict[str, float | str]] = []
    period_frames = [("overall", "all", frame)]
    for period_type in ("week", "month"):
        period_frames.extend(
            (period_type, str(period), group)
            for period, group in frame.groupby(period_type, sort=True)
        )

    for period_type, period, period_frame in period_frames:
        groups = [("all", period_frame)]
        groups.extend(
            (str(model_type), group)
            for model_type, group in period_frame.groupby(
                "model_type", sort=True, dropna=False
            )
        )
        for model_type, group in groups:
            reports.append(
                {
                    "period_type": period_type,
                    "period": period,
                    "model_type": model_type,
                    **_group_metrics(
                        group,
                        baseline_probability=baseline_probability,
                    ),
                }
            )
    report = pd.DataFrame(reports)
    report["operating_decision"] = ""
    overall_mask = (
        report["period_type"].eq("overall")
        & report["model_type"].eq("all")
    )
    overall = report.loc[overall_mask].iloc[0]
    monthly = report.loc[
        report["period_type"].eq("month")
        & report["model_type"].eq("all")
    ].sort_values("period")
    monthly_fallback_rate = float(monthly.iloc[-1]["fallback_rate"])

    latest_month = frame["month"].max()
    current_prior_rate = pd.to_numeric(
        frame.loc[
            frame["month"].eq(latest_month),
            "feature_prior_usage_rate",
        ],
        errors="coerce",
    ).mean()
    latest_month_start = pd.Timestamp(f"{latest_month}-01", tz="Asia/Seoul")
    prior_start = latest_month_start - pd.Timedelta(days=28)
    prior_mask = (
        seoul_time.ge(prior_start)
        & seoul_time.lt(latest_month_start)
    )
    previous_prior_rate = pd.to_numeric(
        frame.loc[prior_mask, "feature_prior_usage_rate"],
        errors="coerce",
    ).mean()
    prior_usage_increase = (
        float(current_prior_rate - previous_prior_rate)
        if pd.notna(previous_prior_rate)
        else 0.0
    )
    decision_metrics = {
        **overall.to_dict(),
        "monthly_fallback_rate": monthly_fallback_rate,
        "prior_usage_increase": prior_usage_increase,
    }
    report.loc[overall_mask, "operating_decision"] = operating_decision(
        decision_metrics
    )
    return report


def calibration_bins(rows: pd.DataFrame) -> pd.DataFrame:
    """모델 유형별 10구간 평균 확률과 실제 홈 승률을 계산한다."""

    if rows.empty:
        return pd.DataFrame()
    frame = rows.copy()
    frame["probability_bin"] = pd.cut(
        pd.to_numeric(frame["home_win_probability"], errors="raise"),
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
    )
    return (
        frame.groupby(["model_type", "probability_bin"], observed=True)
        .agg(
            n_games=("s_no", "size"),
            mean_probability=("home_win_probability", "mean"),
            actual_home_win_rate=("target_home_win", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    log_path = OPERATIONS_DIR / "prediction_log.csv"
    if not log_path.exists():
        print("prediction_log.csv가 없어 평가를 건너뜁니다.")
        return

    log = pd.read_csv(log_path)
    games = pd.read_csv(RAW_DATA_DIR / "games_master.csv")
    metadata = json.loads(
        (MODEL_DIR / "best_model_metadata.json").read_text(encoding="utf-8")
    )
    baseline_probability = float(metadata["baseline_home_probability"])
    deployment_ids = (
        log.loc[
            log["record_type"].eq("prediction")
            & log["evaluation_role"].eq("prospective_holdout"),
            "deployment_id",
        ]
        .dropna()
        .astype(str)
        .unique()
    )
    reports = []
    bins = []
    for deployment_id in sorted(deployment_ids):
        rows = prepare_evaluation_rows(
            log,
            games,
            deployment_id=deployment_id,
        )
        report = evaluate_prediction_log(rows, baseline_probability)
        if not report.empty:
            report.insert(0, "deployment_id", deployment_id)
            reports.append(report)
        bin_report = calibration_bins(rows)
        if not bin_report.empty:
            bin_report.insert(0, "deployment_id", deployment_id)
            bins.append(bin_report)

    if not reports:
        print("평가 가능한 전향적 예측이 없습니다.")
        return
    report = pd.concat(reports, ignore_index=True)
    report.to_csv(
        EVALUATIONS_DIR / "prediction_performance_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(bins, ignore_index=True).to_csv(
        EVALUATIONS_DIR / "prediction_calibration_bins.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
