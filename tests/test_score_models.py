import unittest

import numpy as np
import pandas as pd


class ScoreModelTests(unittest.TestCase):
    def test_conditional_skellam_probability_is_bounded_and_symmetric(self):
        # 무승부 확률을 버린 뒤 재정규화하지 않는 회귀를 막는다.
        from score_models import conditional_skellam_home_probability

        equal = conditional_skellam_home_probability(
            np.array([4.0]), np.array([4.0])
        )
        favored = conditional_skellam_home_probability(
            np.array([6.0]), np.array([3.0])
        )

        self.assertAlmostEqual(equal[0], 0.5)
        self.assertGreater(favored[0], 0.5)
        self.assertTrue(((favored >= 0) & (favored <= 1)).all())

    def test_negative_score_means_are_clipped_before_probability_conversion(self):
        # 회귀기의 음수 기대득점이 분포 계산을 깨뜨리는 회귀를 막는다.
        from score_models import conditional_skellam_home_probability

        probability = conditional_skellam_home_probability(
            np.array([-2.0]), np.array([1.0])
        )

        self.assertTrue(np.isfinite(probability[0]))
        self.assertTrue(0 <= probability[0] <= 1)

    def test_score_fold_calibration_is_separate_from_validation(self):
        from train_score_models import _evaluate_score_fold

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
                "homeScore": [2, 5, 1, 6, 1],
                "awayScore": [3, 1, 4, 2, 5],
                "homeTeam": [1, 2, 1, 2, 1],
                "awayTeam": [2, 1, 2, 1, 2],
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
                "homeScore": [3, 1],
                "awayScore": [2, 5],
                "homeTeam": [1, 2],
                "awayTeam": [2, 1],
            },
            index=[10, 11],
        )

        val_frame, probs, val_home, val_away = _evaluate_score_fold(
            "poisson_catboost", ["feature"], train, validation
        )

        self.assertEqual(list(val_frame["s_no"]), [5, 6])
        self.assertTrue(((probs >= 0) & (probs <= 1)).all())


if __name__ == "__main__":
    unittest.main()
