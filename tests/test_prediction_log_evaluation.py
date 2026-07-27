import math
import unittest

import pandas as pd


def _prediction_log() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_type": [
                "prediction",
                "prediction",
                "prediction",
                "prediction",
                "delivery",
                "delivery",
            ],
            "s_no": [1, 2, 3, 4, 1, 2],
            "recorded_at": [
                "2026-07-28T17:00:00+09:00",
                "2026-07-28T17:00:00+09:00",
                "2026-07-28T17:00:00+09:00",
                "2026-07-29T19:00:00+09:00",
                "2026-07-28T17:01:00+09:00",
                "2026-07-28T17:01:00+09:00",
            ],
            "game_datetime": [
                "2026-07-28T18:30:00+09:00",
                "2026-07-28T18:30:00+09:00",
                "2026-07-28T18:30:00+09:00",
                "2026-07-29T18:30:00+09:00",
                "2026-07-28T18:30:00+09:00",
                "2026-07-28T18:30:00+09:00",
            ],
            "home_win_probability": [0.8, 0.2, 0.7, 0.6, 0.8, 0.2],
            "model_type": [
                "primary",
                "fallback_recent10",
                "primary",
                "primary",
                "primary",
                "fallback_recent10",
            ],
            "deployment_id": [
                "deploy-a",
                "deploy-a",
                "deploy-b",
                "deploy-a",
                "deploy-a",
                "deploy-a",
            ],
            "evaluation_role": [
                "prospective_holdout",
                "prospective_holdout",
                "prospective_holdout",
                "prospective_holdout",
                "prospective_holdout",
                "prospective_holdout",
            ],
            "prospective_start_date": ["2026-07-28"] * 6,
            "lineup_complete": [True, False, True, True, True, False],
            "feature_prior_usage_rate": [0.0, 0.5, 0.0, 0.0, 0.0, 0.5],
            "api_status": [
                "pending",
                "pending",
                "pending",
                "pending",
                "success",
                "failed",
            ],
        }
    )


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "s_no": [1, 2, 3, 4],
            "homeScore": [3, 1, 4, 5],
            "awayScore": [2, 2, 1, 1],
        }
    )


class PredictionLogEvaluationTests(unittest.TestCase):
    def test_only_registered_pregame_prospective_deployment_is_evaluated(self):
        # 다른 배포와 경기 시작 이후 예측이 전향적 표본에 섞이면 안 된다.
        from evaluate_prediction_log import prepare_evaluation_rows

        result = prepare_evaluation_rows(
            _prediction_log(),
            _games(),
            deployment_id="deploy-a",
        )

        self.assertEqual(result["s_no"].tolist(), [1, 2])
        self.assertEqual(result["delivery_success"].tolist(), [1.0, 0.0])

    def test_engineering_regression_rows_are_excluded(self):
        from evaluate_prediction_log import prepare_evaluation_rows

        log = _prediction_log()
        log.loc[log["s_no"].eq(2), "evaluation_role"] = "engineering_regression"

        result = prepare_evaluation_rows(
            log,
            _games(),
            deployment_id="deploy-a",
        )

        self.assertEqual(result["s_no"].tolist(), [1])

    def test_overall_rates_use_all_predictions(self):
        # 모델 유형별 분리 때문에 전체 대체 모델·API 성공률이 0 또는 1로 고정되면 안 된다.
        from evaluate_prediction_log import (
            evaluate_prediction_log,
            prepare_evaluation_rows,
        )

        rows = prepare_evaluation_rows(
            _prediction_log(),
            _games(),
            deployment_id="deploy-a",
        )
        report = evaluate_prediction_log(rows, baseline_probability=0.52)
        overall = report.loc[
            report["period_type"].eq("overall")
            & report["model_type"].eq("all")
        ].iloc[0]

        self.assertEqual(overall["fallback_rate"], 0.5)
        self.assertEqual(overall["api_success_rate"], 0.5)
        self.assertEqual(overall["lineup_completion_rate"], 0.5)
        self.assertEqual(overall["feature_prior_usage_rate"], 0.25)
        self.assertTrue(math.isnan(overall["roc_auc"]))
        self.assertEqual(overall["operating_decision"], "추가 관찰")

    def test_bootstrap_is_reproducible_and_delta_is_model_minus_baseline(self):
        from evaluate_prediction_log import paired_day_block_bootstrap

        rows = pd.DataFrame(
            {
                "game_datetime": pd.to_datetime(
                    [
                        "2026-07-28T18:30:00+09:00",
                        "2026-07-29T18:30:00+09:00",
                    ],
                    utc=True,
                ),
                "target_home_win": [1, 0],
                "home_win_probability": [0.9, 0.1],
            }
        )

        first = paired_day_block_bootstrap(
            rows,
            0.5,
            iterations=100,
            seed=42,
        )
        second = paired_day_block_bootstrap(
            rows,
            0.5,
            iterations=100,
            seed=42,
        )

        self.assertEqual(first, second)
        self.assertLess(first["mean_log_loss_delta"], 0)
        self.assertLess(first["mean_brier_delta"], 0)

    def test_operating_decision_uses_fixed_priority(self):
        from evaluate_prediction_log import operating_decision

        metrics = {
            "n_games": 300,
            "monthly_fallback_rate": 0.25,
            "prior_usage_increase": 0.20,
            "log_loss_delta_ci_low": 0.01,
            "log_loss_delta_ci_high": 0.03,
            "brier_delta_ci_low": 0.01,
            "brier_delta_ci_high": 0.02,
            "calibration_intercept": 0.2,
            "calibration_slope": 0.6,
        }

        self.assertEqual(operating_decision(metrics), "데이터 파이프라인 점검")
        metrics["monthly_fallback_rate"] = 0.1
        metrics["prior_usage_increase"] = 0.0
        self.assertEqual(operating_decision(metrics), "모델 재검토")
        metrics["log_loss_delta_ci_low"] = -0.02
        self.assertEqual(operating_decision(metrics), "재보정 검토")

    def test_small_or_single_class_segment_keeps_report_with_nan_metrics(self):
        from evaluate_prediction_log import evaluate_prediction_log

        rows = pd.DataFrame(
            {
                "s_no": [1, 2],
                "game_datetime": pd.to_datetime(
                    [
                        "2026-07-28T18:30:00+09:00",
                        "2026-07-29T18:30:00+09:00",
                    ],
                    utc=True,
                ),
                "home_win_probability": [0.6, 0.7],
                "target_home_win": [1, 1],
                "model_type": ["primary", "primary"],
                "delivery_success": [1.0, 1.0],
                "lineup_complete": [True, True],
                "feature_prior_usage_rate": [0.0, 0.0],
            }
        )

        report = evaluate_prediction_log(rows, baseline_probability=0.52)
        overall = report.loc[
            report["period_type"].eq("overall")
            & report["model_type"].eq("all")
        ].iloc[0]

        self.assertTrue(math.isnan(overall["calibration_intercept"]))
        self.assertTrue(math.isnan(overall["calibration_slope"]))
        self.assertTrue(math.isnan(overall["roc_auc"]))


if __name__ == "__main__":
    unittest.main()
