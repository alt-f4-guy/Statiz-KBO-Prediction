import unittest

import pandas as pd


class ModelComparisonTests(unittest.TestCase):
    def test_operating_metadata_uses_relative_paths_and_frozen_baseline(self):
        # 배포 메타데이터가 로컬 절대 경로와 가변 기준선에 의존하면 안 된다.
        from compare_models import build_operating_metadata

        metadata = build_operating_metadata(
            selected="catboost",
            family="direct_classifier",
            baseline_probability=0.5184,
        )

        self.assertEqual(
            metadata["model_path"],
            "artifacts/models/best_model.joblib",
        )
        self.assertEqual(
            metadata["split_manifest"],
            "data/final/time_split_manifest.json",
        )
        self.assertEqual(metadata["baseline_home_probability"], 0.5184)

    def test_models_must_evaluate_identical_game_ids(self):
        # 서로 다른 경기 집합의 지표를 직접 비교하는 오류를 막는다.
        from compare_models import assert_same_evaluation_games

        predictions = pd.DataFrame(
            {
                "model": ["a", "a", "b", "b"],
                "s_no": [1, 2, 1, 3],
            }
        )

        with self.assertRaisesRegex(ValueError, "동일하지 않습니다"):
            assert_same_evaluation_games(predictions)

    def test_classifier_is_selected_when_development_metrics_win(self):
        # 정확도만으로 모델을 선택하는 회귀를 막는다.
        from compare_models import select_operating_model

        summary = pd.DataFrame(
            {
                "model": ["classifier", "score"],
                "family": ["direct_classifier", "score_distribution"],
                "development_log_loss": [0.66, 0.68],
                "development_brier_score": [0.23, 0.24],
                "development_calibration_intercept": [0.02, 0.01],
                "development_calibration_slope": [0.98, 1.01],
                "final_log_loss": [0.67, 0.69],
                "final_brier_score": [0.24, 0.25],
                "final_calibration_intercept": [0.02, 0.01],
                "final_calibration_slope": [0.98, 1.01],
                "accuracy": [0.51, 0.60],
            }
        )

        selected = select_operating_model(summary)

        self.assertEqual(selected, "classifier")

    def test_final_test_metrics_do_not_select_operating_model(self):
        from compare_models import select_operating_model

        summary = pd.DataFrame(
            {
                "model": ["classifier", "score"],
                "family": ["direct_classifier", "score_distribution"],
                "development_log_loss": [0.66, 0.68],
                "development_brier_score": [0.23, 0.24],
                "development_calibration_intercept": [0.02, 0.01],
                "development_calibration_slope": [0.98, 1.01],
                "final_log_loss": [9.0, 0.01],
                "final_brier_score": [0.90, 0.01],
                "final_calibration_intercept": [8.0, 0.0],
                "final_calibration_slope": [8.0, 1.0],
            }
        )

        selected = select_operating_model(summary)

        self.assertEqual(selected, "classifier")


if __name__ == "__main__":
    unittest.main()
