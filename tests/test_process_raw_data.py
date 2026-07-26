import unittest

import numpy as np
import pandas as pd


class RawDataProcessingTests(unittest.TestCase):
    def test_latest_successful_snapshot_wins_over_older_success_and_newer_error(self):
        # 오류 응답 또는 오래된 정상 응답을 선택하는 중복 제거 회귀를 막는다.
        from raw_data_processing import select_latest_successful_snapshots

        snapshots = pd.DataFrame(
            {
                "p_no": [10, 10, 10],
                "year_req": [2026, 2026, 2026],
                "fetched_at": [
                    "2026-07-01T10:00:00+09:00",
                    "2026-07-02T10:00:00+09:00",
                    "2026-07-03T10:00:00+09:00",
                ],
                "response_status": ["success", "success", "error"],
                "json": ["old", "latest", "failure"],
            }
        )

        result = select_latest_successful_snapshots(
            snapshots, ["p_no", "year_req"]
        )

        self.assertEqual(result.iloc[0]["json"], "latest")
        self.assertEqual(len(result), 1)

    def test_baseball_innings_are_converted_vectorially(self):
        # 0.1/0.2를 십진 이닝으로 해석하는 오류를 막는다.
        from raw_data_processing import convert_baseball_innings

        result = convert_baseball_innings(pd.Series(["5.1", "5.2", "6", None]))

        np.testing.assert_allclose(
            result.iloc[:3].to_numpy(),
            np.array([5 + 1 / 3, 5 + 2 / 3, 6.0]),
        )
        self.assertTrue(np.isnan(result.iloc[3]))

    def test_player_game_duplicates_keep_latest_snapshot_value(self):
        # 같은 선수-경기가 여러 스냅샷에 있을 때 오래된 기록이 남으면 안 된다.
        from raw_data_processing import deduplicate_player_games

        games = pd.DataFrame(
            {
                "p_no": [10, 10],
                "s_no_key": [20260001, 20260001],
                "fetched_at": [
                    "2026-07-01T10:00:00+09:00",
                    "2026-07-02T10:00:00+09:00",
                ],
                "H": [1, 2],
            }
        )

        result = deduplicate_player_games(games)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["H"], 2)


if __name__ == "__main__":
    unittest.main()
