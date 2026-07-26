import unittest

import pandas as pd


class FinalDatasetContractTests(unittest.TestCase):
    def test_duplicate_game_id_is_rejected(self):
        # 동일 경기를 중복 학습시키는 데이터 계약 위반을 막는다.
        from dataset_contract import DatasetContractError, validate_final_dataset

        data = self._valid_rows()
        data.loc[1, "s_no"] = data.loc[0, "s_no"]

        with self.assertRaisesRegex(DatasetContractError, "s_no 중복"):
            validate_final_dataset(data)

    def test_future_feature_cutoff_is_rejected(self):
        # 경기 시작 이후 정보가 피처에 들어가는 시점 누수를 막는다.
        from dataset_contract import DatasetContractError, validate_final_dataset

        data = self._valid_rows()
        data.loc[0, "feature_cutoff_datetime"] = data.loc[0, "game_datetime"]

        with self.assertRaisesRegex(DatasetContractError, "기준시각"):
            validate_final_dataset(data)

    def test_prediction_mode_keeps_unscored_schedule_row(self):
        # 실시간 목표 경기가 점수 미입력이라는 이유로 피처 생성에서 사라지는 회귀를 막는다.
        from feature_matrix_v9 import _prepare_games

        games = pd.DataFrame(
            {
                "s_no": [1],
                "gameDate": [1751169600],
                "gameDateResume": [0],
                "homeTeam": [1],
                "awayTeam": [2],
                "homeScore": [pd.NA],
                "awayScore": [pd.NA],
            }
        )

        result = _prepare_games(games, include_unscored=True)

        self.assertEqual(result["s_no"].tolist(), [1])

    @staticmethod
    def _valid_rows():
        return pd.DataFrame(
            {
                "s_no": [1, 2],
                "game_datetime": pd.to_datetime(
                    [
                        "2026-04-01T18:30:00+09:00",
                        "2026-04-02T18:30:00+09:00",
                    ]
                ),
                "feature_cutoff_datetime": pd.to_datetime(
                    [
                        "2026-04-01T18:29:59+09:00",
                        "2026-04-02T18:29:59+09:00",
                    ]
                ),
                "year": [2026, 2026],
                "homeTeam": [1, 2],
                "awayTeam": [2, 1],
                "homeScore": [3, 2],
                "awayScore": [2, 1],
                "home_sp_source": ["prior_season", "current_season"],
                "away_sp_source": ["prior_season", "current_season"],
            }
        )


if __name__ == "__main__":
    unittest.main()
