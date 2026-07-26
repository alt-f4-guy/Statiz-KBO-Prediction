import unittest

import pandas as pd


class SabermetricsTests(unittest.TestCase):
    def test_fip_constant_is_derived_separately_by_year(self):
        # 고정 3.10 상수로 되돌아가는 회귀를 막는다.
        from sabermetrics import calculate_kbo_year_constants

        pitching = pd.DataFrame(
            {
                "year": [2024, 2024, 2025, 2025],
                "IP": [9.0, 9.0, 9.0, 9.0],
                "ER": [3, 3, 5, 5],
                "HR": [1, 1, 1, 1],
                "BB": [2, 2, 2, 2],
                "HP": [0, 0, 0, 0],
                "SO": [8, 8, 4, 4],
            }
        )

        result = calculate_kbo_year_constants(pitching)

        constants = result.set_index("year")["fip_constant"]
        self.assertNotEqual(constants.loc[2024], constants.loc[2025])
        self.assertAlmostEqual(constants.loc[2024], 3.0 - (6 / 18))
        self.assertAlmostEqual(constants.loc[2025], 5.0 - (22 / 18))

    def test_park_factor_for_a_game_ignores_that_game_and_future_results(self):
        # 평가 경기 이후 득점을 구장 팩터에 포함하는 누수를 막는다.
        from sabermetrics import calculate_asof_park_factor

        games = pd.DataFrame(
            {
                "s_no": [1, 2, 3],
                "game_datetime": pd.to_datetime(
                    [
                        "2026-04-01T18:30:00+09:00",
                        "2026-04-02T18:30:00+09:00",
                        "2026-04-03T18:30:00+09:00",
                    ]
                ),
                "feature_cutoff_datetime": pd.to_datetime(
                    [
                        "2026-04-01T18:29:59+09:00",
                        "2026-04-02T18:29:59+09:00",
                        "2026-04-03T18:29:59+09:00",
                    ],
                    utc=True,
                ),
                "result_available_datetime": pd.to_datetime(
                    [
                        "2026-04-01T22:00:00+09:00",
                        "2026-04-02T22:00:00+09:00",
                        "2026-04-03T22:00:00+09:00",
                    ],
                    utc=True,
                ),
                "s_code": [1001, 1001, 1001],
                "homeScore": [5, 4, 99],
                "awayScore": [3, 2, 99],
            }
        )

        original = calculate_asof_park_factor(games)
        changed = games.copy()
        changed.loc[2, ["homeScore", "awayScore"]] = [0, 0]
        revised = calculate_asof_park_factor(changed)

        self.assertEqual(original.loc[1, "park_factor"], revised.loc[1, "park_factor"])

    def test_park_factor_requires_result_availability_before_cutoff(self):
        # 시작했지만 아직 끝나지 않은 경기 점수가 뒤 경기 피처에 들어가는 누수를 막는다.
        from sabermetrics import calculate_asof_park_factor

        games = pd.DataFrame(
            {
                "s_no": [1, 2, 3],
                "game_datetime": pd.to_datetime(
                    [
                        "2026-04-01T18:30:00+09:00",
                        "2026-04-02T14:00:00+09:00",
                        "2026-04-02T17:00:00+09:00",
                    ],
                    utc=True,
                ),
                "feature_cutoff_datetime": pd.to_datetime(
                    [
                        "2026-04-01T18:29:59+09:00",
                        "2026-04-02T13:59:59+09:00",
                        "2026-04-02T16:59:59+09:00",
                    ],
                    utc=True,
                ),
                "result_available_datetime": pd.to_datetime(
                    [
                        "2026-04-01T22:00:00+09:00",
                        "2026-04-02T18:00:00+09:00",
                        "2026-04-02T21:00:00+09:00",
                    ],
                    utc=True,
                ),
                "s_code": [100, 200, 100],
                "homeScore": [5, 1, 3],
                "awayScore": [5, 0, 2],
            }
        )
        changed = games.copy()
        changed.loc[1, ["homeScore", "awayScore"]] = [100, 0]

        original = calculate_asof_park_factor(games).set_index("s_no")
        revised = calculate_asof_park_factor(changed).set_index("s_no")

        self.assertAlmostEqual(original.loc[3, "park_factor"], 1.0)
        self.assertAlmostEqual(revised.loc[3, "park_factor"], 1.0)

    def test_first_available_year_has_explicit_league_batting_prior(self):
        # 이전 시즌 파일이 없는 첫 학습 연도의 타격 피처 전체 결측을 막는다.
        from feature_matrix_v9 import _league_batting_priors

        season = pd.DataFrame(
            {
                "p_no": [10],
                "year": [2023],
                "PA": [100],
                "AB": [90],
                "H": [25],
                "1B": [20],
                "2B": [4],
                "3B": [0],
                "HR": [1],
                "TB": [32],
                "BB": [8],
                "HP": [1],
                "SO": [20],
                "SF": [1],
            }
        )
        constants = pd.DataFrame(
            {
                "year": [2023, 2024],
                "weight_bb": [0.69, 0.69],
                "weight_hp": [0.72, 0.72],
                "weight_1b": [0.88, 0.88],
                "weight_2b": [1.247, 1.247],
                "weight_3b": [1.578, 1.578],
                "weight_hr": [2.031, 2.031],
            }
        )

        result = _league_batting_priors(season, constants)

        first_year = result.loc[result["year"].eq(2023)].iloc[0]
        self.assertAlmostEqual(first_year["league_obp"], 0.33)
        self.assertFalse(first_year.isna().any())

    def test_asof_constants_ignore_current_and_future_events(self):
        from sabermetrics import calculate_asof_kbo_constants

        games = pd.DataFrame(
            {
                "s_no": [1, 2],
                "year": [2026, 2026],
                "feature_cutoff_datetime": pd.to_datetime(
                    [
                        "2026-04-02T18:29:59+09:00",
                        "2026-04-03T18:29:59+09:00",
                    ],
                    utc=True,
                ),
            }
        )
        pitching = pd.DataFrame(
            {
                "year": [2026, 2026],
                "event_datetime": pd.to_datetime(
                    [
                        "2026-04-01T18:30:00+09:00",
                        "2026-04-03T18:30:00+09:00",
                    ],
                    utc=True,
                ),
                "IP": [9.0, 9.0],
                "ER": [3.0, 99.0],
                "HR": [1.0, 20.0],
                "BB": [2.0, 30.0],
                "HP": [0.0, 10.0],
                "SO": [8.0, 0.0],
            }
        )
        batting = pd.DataFrame(
            {
                "year": [2026, 2026],
                "event_datetime": pitching["event_datetime"],
                "R": [4.0, 99.0],
                "PA": [36.0, 36.0],
            }
        )

        original = calculate_asof_kbo_constants(games, pitching, batting)
        changed_pitching = pitching.copy()
        changed_batting = batting.copy()
        changed_pitching.loc[1, ["ER", "HR", "BB"]] = [0.0, 0.0, 0.0]
        changed_batting.loc[1, "R"] = 0.0
        revised = calculate_asof_kbo_constants(
            games, changed_pitching, changed_batting
        )

        pd.testing.assert_series_equal(
            original.loc[0],
            revised.loc[0],
            check_names=False,
        )


if __name__ == "__main__":
    unittest.main()
