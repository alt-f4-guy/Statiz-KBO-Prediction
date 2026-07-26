import unittest

import pandas as pd


class GameDatetimeTests(unittest.TestCase):
    def test_unix_timestamp_is_converted_to_seoul_time(self):
        # UTC로 해석하거나 로컬 시스템 시간대를 사용하는 회귀를 막는다.
        from game_time import build_game_datetime_reference

        games = pd.DataFrame(
            {
                "s_no": [20230001],
                "gameDate": [1680325200],
                "gameDateResume": [0],
            }
        )

        result = build_game_datetime_reference(games)

        self.assertEqual(
            result.loc[0, "game_datetime"].isoformat(),
            "2023-04-01T14:00:00+09:00",
        )
        self.assertLess(
            result.loc[0, "feature_cutoff_datetime"],
            result.loc[0, "game_datetime"],
        )

    def test_doubleheader_second_game_is_frozen_before_first_game(self):
        # 2차전 피처에 1차전 결과가 들어가는 시점 누수를 막는다.
        from game_time import build_game_datetime_reference

        games = pd.DataFrame(
            {
                "s_no": [1, 2],
                "gameDate": [1751169600, 1751180400],
                "gameDateResume": [0, 0],
                "homeTeam": [1001, 1001],
                "awayTeam": [2002, 2002],
            }
        )

        result = build_game_datetime_reference(games)

        self.assertEqual(
            result.loc[1, "feature_cutoff_datetime"],
            result.loc[0, "feature_cutoff_datetime"],
        )

    def test_resume_timestamp_does_not_move_feature_cutoff(self):
        # 서스펜디드 경기 재개 시점의 정보를 원래 경기에 추가하는 누수를 막는다.
        from game_time import build_game_datetime_reference

        games = pd.DataFrame(
            {
                "s_no": [1],
                "gameDate": [1751169600],
                "gameDateResume": [1751256000],
            }
        )

        result = build_game_datetime_reference(games)

        expected = result.loc[0, "game_datetime"] - pd.Timedelta(microseconds=1)
        self.assertEqual(result.loc[0, "feature_cutoff_datetime"], expected)

    def test_result_available_datetime_contracts(self):
        from game_time import build_game_datetime_reference

        games = pd.DataFrame(
            {
                "s_no": [1, 2, 3],
                "gameDate": [1680325200, 1680325200, 1680325200],
                "gameDateResume": [0, 0, 1680411600],
                "result_observed_at": ["2023-04-01T18:15:00+09:00", pd.NA, pd.NA],
                "homeScore": [3, 4, 5],
                "awayScore": [1, 2, 3],
            }
        )

        result = build_game_datetime_reference(games)

        self.assertEqual(
            result.loc[0, "result_available_datetime"].isoformat(),
            "2023-04-01T18:15:00+09:00",
        )
        self.assertEqual(
            result.loc[1, "result_available_datetime"].isoformat(),
            "2023-04-02T00:00:00+09:00",
        )
        self.assertEqual(
            result.loc[2, "result_available_datetime"].isoformat(),
            "2023-04-03T00:00:00+09:00",
        )


if __name__ == "__main__":
    unittest.main()
