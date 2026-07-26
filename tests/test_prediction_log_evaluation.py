import unittest

import pandas as pd


class PredictionLogEvaluationTests(unittest.TestCase):
    def test_predictions_created_after_game_start_are_excluded(self):
        # 사후 생성 예측이 운영 성능에 포함되는 회귀를 막는다.
        from evaluate_prediction_log import prepare_evaluation_rows

        log = pd.DataFrame(
            {
                "record_type": ["prediction", "prediction"],
                "s_no": [1, 2],
                "recorded_at": [
                    "2026-04-01T17:00:00+09:00",
                    "2026-04-02T19:00:00+09:00",
                ],
                "game_datetime": [
                    "2026-04-01T18:30:00+09:00",
                    "2026-04-02T18:30:00+09:00",
                ],
                "home_win_probability": [0.6, 0.7],
                "model_type": ["primary", "primary"],
            }
        )
        games = pd.DataFrame(
            {
                "s_no": [1, 2],
                "homeScore": [3, 4],
                "awayScore": [2, 1],
            }
        )

        result = prepare_evaluation_rows(log, games)

        self.assertEqual(result["s_no"].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
