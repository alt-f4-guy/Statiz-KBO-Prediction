import unittest

import pandas as pd


class RecentTenFallbackTests(unittest.TestCase):
    def test_probability_uses_only_games_before_feature_cutoff(self):
        # 목표 경기 이후 또는 더블헤더 1차전 결과가 2차전 대체 확률에 들어가는 누수를 막는다.
        from fallback_recent10 import recent_ten_home_probability

        history = pd.DataFrame(
            {
                "game_datetime": pd.to_datetime(
                    [
                        "2026-04-01T14:00:00+09:00",
                        "2026-04-02T14:00:00+09:00",
                        "2026-04-03T14:00:00+09:00",
                        "2026-04-04T14:00:00+09:00",
                        "2026-04-05T14:00:00+09:00",
                        "2026-04-06T14:00:00+09:00",
                        "2026-04-07T14:00:00+09:00",
                        "2026-04-08T14:00:00+09:00",
                        "2026-04-09T14:00:00+09:00",
                        "2026-04-10T14:00:00+09:00",
                        "2026-04-11T14:00:00+09:00",
                    ]
                ),
                "result_available_datetime": pd.to_datetime(
                    [
                        "2026-04-01T18:00:00+09:00",
                        "2026-04-02T18:00:00+09:00",
                        "2026-04-03T18:00:00+09:00",
                        "2026-04-04T18:00:00+09:00",
                        "2026-04-05T18:00:00+09:00",
                        "2026-04-06T18:00:00+09:00",
                        "2026-04-07T18:00:00+09:00",
                        "2026-04-08T18:00:00+09:00",
                        "2026-04-09T18:00:00+09:00",
                        "2026-04-10T18:00:00+09:00",
                        "2026-04-11T18:00:00+09:00",
                    ]
                ),
                "homeTeam": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
                "awayTeam": [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                "homeScore": [3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 99],
                "awayScore": [1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 0],
            }
        )
        cutoff = pd.Timestamp("2026-04-11T13:59:59+09:00")

        probability = recent_ten_home_probability(
            history,
            home_team=1,
            away_team=2,
            feature_cutoff_datetime=cutoff,
            league_home_win_rate=0.52,
        )

        self.assertAlmostEqual(probability, 0.65)

    def test_less_than_five_games_uses_league_home_rate(self):
        # 시즌 초 소표본 승률을 과신하는 회귀를 막는다.
        from fallback_recent10 import recent_ten_home_probability

        history = pd.DataFrame(
            {
                "game_datetime": pd.to_datetime(
                    ["2026-04-01T14:00:00+09:00"]
                ),
                "result_available_datetime": pd.to_datetime(
                    ["2026-04-01T18:00:00+09:00"]
                ),
                "homeTeam": [1],
                "awayTeam": [2],
                "homeScore": [3],
                "awayScore": [1],
            }
        )

        probability = recent_ten_home_probability(
            history,
            home_team=1,
            away_team=2,
            feature_cutoff_datetime=pd.Timestamp(
                "2026-04-02T14:00:00+09:00"
            ),
            league_home_win_rate=0.54,
        )

        self.assertEqual(probability, 0.54)

    def test_recent_probability_requires_result_availability_before_cutoff(self):
        from fallback_recent10 import recent_ten_home_probability

        history = pd.DataFrame(
            {
                "game_datetime": pd.to_datetime(
                    [
                        "2026-04-01T18:30:00+09:00",
                        "2026-04-02T18:30:00+09:00",
                        "2026-04-03T18:30:00+09:00",
                        "2026-04-04T18:30:00+09:00",
                        "2026-04-05T18:30:00+09:00",
                        "2026-04-06T18:30:00+09:00",
                    ],
                    utc=True,
                ),
                "result_available_datetime": pd.to_datetime(
                    [
                        "2026-04-01T22:30:00+09:00",
                        "2026-04-02T22:30:00+09:00",
                        "2026-04-03T22:30:00+09:00",
                        "2026-04-04T22:30:00+09:00",
                        "2026-04-05T22:30:00+09:00",
                        "2026-04-06T22:30:00+09:00",
                    ],
                    utc=True,
                ),
                "homeTeam": [1, 1, 1, 1, 1, 1],
                "awayTeam": [2, 2, 2, 2, 2, 2],
                "homeScore": [3.0, 3.0, 3.0, 1.0, 1.0, 1.0],
                "awayScore": [1.0, 1.0, 1.0, 3.0, 3.0, 3.0],
            }
        )

        probability = recent_ten_home_probability(
            history,
            home_team=1,
            away_team=2,
            feature_cutoff_datetime=pd.Timestamp(
                "2026-04-06T20:00:00+09:00"
            ),
            league_home_win_rate=0.54,
        )

        self.assertAlmostEqual(probability, 4 / 7)


if __name__ == "__main__":
    unittest.main()
