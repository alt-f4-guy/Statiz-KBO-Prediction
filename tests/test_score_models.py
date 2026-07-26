import unittest

import numpy as np


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


if __name__ == "__main__":
    unittest.main()
