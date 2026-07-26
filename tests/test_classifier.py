import unittest

import numpy as np
import pandas as pd


class ClassifierTests(unittest.TestCase):
    def test_draws_are_excluded_and_target_is_home_win(self):
        # 무승부를 원정 승리로 잘못 학습하는 목표 정의 회귀를 막는다.
        from classifier_model import prepare_classifier_frame

        data = pd.DataFrame(
            {
                "s_no": [1, 2, 3],
                "homeScore": [3, 2, 1],
                "awayScore": [2, 2, 4],
                "homeTeam": [10, 20, 30],
                "awayTeam": [20, 30, 10],
                "feature": [0.1, 0.2, 0.3],
            }
        )

        prepared = prepare_classifier_frame(data)

        self.assertEqual(prepared["s_no"].tolist(), [1, 3])
        self.assertEqual(prepared["target_home_win"].tolist(), [1, 0])

    def test_logistic_classifier_probabilities_are_valid_and_reproducible(self):
        # 시드 또는 확률 열 순서가 바뀌는 회귀를 막는다.
        from classifier_model import build_classifier, select_model_features

        data = pd.DataFrame(
            {
                "homeTeam": [1, 2, 1, 2, 1, 2],
                "awayTeam": [2, 1, 2, 1, 2, 1],
                "feature": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
                "target_home_win": [0, 0, 0, 1, 1, 1],
            }
        )
        columns = select_model_features(data)
        first = build_classifier("logistic", columns)
        second = build_classifier("logistic", columns)
        first.fit(data[columns], data["target_home_win"])
        second.fit(data[columns], data["target_home_win"])

        first_probability = first.predict_proba(data[columns])
        second_probability = second.predict_proba(data[columns])

        np.testing.assert_allclose(first_probability.sum(axis=1), 1.0)
        np.testing.assert_allclose(first_probability, second_probability)
        self.assertTrue(((first_probability >= 0) & (first_probability <= 1)).all())

    def test_classifier_fold_calibration_is_separate_from_validation(self):
        from train_classifier import _evaluate_classifier_fold

        dates = pd.to_datetime(
            [
                "2025-04-01T18:30:00+09:00",
                "2025-04-02T18:30:00+09:00",
                "2025-04-03T18:30:00+09:00",
                "2025-04-04T18:30:00+09:00",
                "2025-04-04T18:30:00+09:00",
            ],
            utc=True,
        )
        train = pd.DataFrame(
            {
                "s_no": [1, 2, 3, 4, 5],
                "game_datetime": dates,
                "feature": [-2.0, -1.0, 1.0, 2.0, -0.5],
                "target_home_win": [0, 1, 0, 1, 0],
            }
        )
        validation = pd.DataFrame(
            {
                "s_no": [5, 6],
                "game_datetime": pd.to_datetime(
                    ["2025-04-05T18:30:00+09:00", "2025-04-06T18:30:00+09:00"],
                    utc=True,
                ),
                "feature": [0.0, 1.5],
                "target_home_win": [0, 1],
            },
            index=[10, 11],
        )

        probs = _evaluate_classifier_fold("logistic", ["feature"], train, validation)

        self.assertEqual(list(probs.index), [10, 11])
        self.assertTrue(((probs >= 0) & (probs <= 1)).all())

    def test_sigmoid_calibrator_uses_unregularized_logistic_regression(self):
        from classifier_model import SigmoidCalibrator

        calibrator = SigmoidCalibrator()

        self.assertTrue(np.isinf(calibrator.model.C))
        self.assertEqual(calibrator.model.random_state, 42)

    def test_probability_metrics_reports_unregularized_calibration_slope(self):
        from classifier_model import probability_metrics, _unregularized_logistic

        rng = np.random.default_rng(42)
        logits = rng.normal(0, 1, size=200)
        probs = 1 / (1 + np.exp(-logits))
        targets = rng.binomial(1, probs)

        metrics = probability_metrics(targets, probs)
        expected = _unregularized_logistic().fit(
            np.log(probs / (1 - probs)).reshape(-1, 1), targets
        )

        self.assertAlmostEqual(
            metrics["calibration_intercept"],
            expected.intercept_[0],
            places=10,
        )
        self.assertAlmostEqual(
            metrics["calibration_slope"],
            expected.coef_[0, 0],
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
