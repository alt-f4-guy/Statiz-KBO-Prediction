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


if __name__ == "__main__":
    unittest.main()
