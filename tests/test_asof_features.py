import unittest

import pandas as pd


class AsofFeatureTests(unittest.TestCase):
    def test_current_game_record_is_excluded_by_cutoff(self):
        # 목표 경기 기록을 선수 누적 피처에 포함하는 시점 누수를 막는다.
        from asof_features import merge_player_asof

        requests = pd.DataFrame(
            {
                "row_id": [1],
                "p_no": [10],
                "feature_cutoff_datetime": pd.to_datetime(
                    ["2026-04-02T18:29:59.999999+09:00"]
                ),
            }
        )
        events = pd.DataFrame(
            {
                "p_no": [10, 10],
                "event_datetime": pd.to_datetime(
                    [
                        "2026-04-01T18:30:00+09:00",
                        "2026-04-02T18:30:00+09:00",
                    ]
                ),
                "cum_pa": [4, 8],
            }
        )

        result = merge_player_asof(requests, events, ["cum_pa"])

        self.assertEqual(result.loc[0, "cum_pa"], 4)

    def test_rotation_pitcher_is_not_a_bullpen_candidate(self):
        # 당일 선발만 제외해 다른 선발 로테이션 투수가 불펜에 섞이는 회귀를 막는다.
        from asof_features import mark_bullpen_candidates

        candidates = pd.DataFrame(
            {
                "p_no": [10, 20, 30, 40],
                "starter_p_no": [10, 10, 10, 10],
                "current_g": [5, 5, 2, 0],
                "current_gs": [5, 0, 1, 0],
                "prior_g": [10, 10, 10, 0],
                "prior_gs": [10, 0, 8, 0],
                "has_pitching_history": [True, True, True, False],
            }
        )

        result = mark_bullpen_candidates(candidates)

        self.assertEqual(
            result["is_bullpen_candidate"].tolist(),
            [False, True, False, False],
        )

    def test_unknown_new_game_id_does_not_control_asof_lookup(self):
        # 신규 s_no가 없다는 이유로 피처가 기본값으로 붕괴하는 회귀를 막는다.
        from asof_features import merge_player_asof

        requests = pd.DataFrame(
            {
                "row_id": [99999999],
                "p_no": [10],
                "feature_cutoff_datetime": pd.to_datetime(
                    ["2026-04-03T18:29:59.999999+09:00"]
                ),
            }
        )
        events = pd.DataFrame(
            {
                "p_no": [10],
                "event_datetime": pd.to_datetime(
                    ["2026-04-01T18:30:00+09:00"]
                ),
                "cum_pa": [4],
            }
        )

        result = merge_player_asof(requests, events, ["cum_pa"])

        self.assertEqual(result.loc[0, "cum_pa"], 4)

    def test_daily_event_year_comes_from_game_datetime(self):
        # year_req만 있는 실제 전처리 스키마에서 연도 결합이 깨지는 회귀를 막는다.
        from feature_matrix_v9 import _prepare_events

        games = pd.DataFrame(
            {
                "s_no": [20260001],
                "game_datetime": pd.to_datetime(
                    ["2026-04-01T18:30:00+09:00"], utc=True
                ),
                "result_available_datetime": pd.to_datetime(
                    ["2026-04-01T22:30:00+09:00"], utc=True
                ),
                "year": [2026],
            }
        )
        day = pd.DataFrame(
            {
                "s_no_key": [20260001],
                "p_no": [10],
                "year_req": [2026],
                "IP": [1.0],
                "PA": [pd.NA],
            }
        )

        pitching, _ = _prepare_events(day, games)

        self.assertEqual(pitching.loc[0, "year"], 2026)

    def test_zero_walk_bullpen_kbb_distinguishes_strikeouts(self):
        from feature_matrix_v9 import _bounded_kbb

        result = _bounded_kbb(
            pd.Series([5.0, 0.0, 8.0]),
            pd.Series([0.0, 0.0, 2.0]),
        )

        self.assertEqual(result.tolist(), [10.0, 2.0, 4.0])

    def test_future_events_do_not_alter_past_features(self):
        from feature_matrix_v9 import _build_asof_constants
        from sabermetrics import add_batting_environment, calculate_kbo_year_constants

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

        prior = add_batting_environment(calculate_kbo_year_constants(pitching), batting)
        orig_asof = _build_asof_constants(games, pitching, batting, prior)

        changed_pitching = pitching.copy()
        changed_batting = batting.copy()
        changed_pitching.loc[1, ["ER", "HR", "BB"]] = [0.0, 0.0, 0.0]
        changed_batting.loc[1, "R"] = 0.0
        rev_asof = _build_asof_constants(games, changed_pitching, changed_batting, prior)

        pd.testing.assert_series_equal(
            orig_asof.loc[0],
            rev_asof.loc[0],
            check_names=False,
        )

    def test_first_year_asof_constants_use_explicit_league_prior(self):
        # 이전 시즌과 완료 경기 기록이 모두 없는 최초 경기의 핵심 피처 결측을 막는다.
        from feature_matrix_v9 import _build_asof_constants
        from sabermetrics import add_batting_environment, calculate_kbo_year_constants

        games = pd.DataFrame(
            {
                "s_no": [1],
                "year": [2023],
                "feature_cutoff_datetime": pd.to_datetime(
                    ["2023-04-01T13:59:59+09:00"],
                    utc=True,
                ),
            }
        )
        pitching = pd.DataFrame(
            {
                "year": [2023],
                "event_datetime": pd.to_datetime(
                    ["2023-04-02T00:00:00+09:00"],
                    utc=True,
                ),
                "IP": [9.0],
                "ER": [4.0],
                "HR": [1.0],
                "BB": [2.0],
                "HP": [0.0],
                "SO": [8.0],
            }
        )
        batting = pd.DataFrame(
            {
                "year": [2023],
                "event_datetime": pitching["event_datetime"],
                "R": [5.0],
                "PA": [40.0],
            }
        )
        prior = add_batting_environment(calculate_kbo_year_constants(pitching), batting)

        result = _build_asof_constants(games, pitching, batting, prior).iloc[0]

        self.assertAlmostEqual(result["league_era"], 4.50)
        self.assertAlmostEqual(result["fip_constant"], 3.10)
        self.assertAlmostEqual(result["league_runs_per_pa"], 0.115)
        self.assertAlmostEqual(result["weight_bb"], 0.69)
        self.assertAlmostEqual(result["weight_hr"], 2.031)

    def test_result_available_datetime_boundaries(self):
        from feature_matrix_v9 import _prepare_events
        from game_time import build_game_datetime_reference

        games = pd.DataFrame(
            {
                "s_no": [1, 2, 3],
                "gameDate": [1680325200, 1680325200, 1680325200],
                "gameDateResume": [0, 0, 1680411600],
                "result_observed_at": ["2023-01-01T18:15:00+09:00", pd.NA, pd.NA],
                "homeScore": [3, 4, 5],
                "awayScore": [1, 2, 3],
                "year": [2023, 2023, 2023],
            }
        )
        ref = build_game_datetime_reference(games)
        ref["year"] = 2023

        day = pd.DataFrame(
            {
                "s_no_key": [1, 2, 3],
                "p_no": [10, 11, 12],
                "year_req": [2023, 2023, 2023],
                "IP": [1.0, 1.0, 1.0],
                "PA": [pd.NA, pd.NA, pd.NA],
            }
        )

        pitching, _ = _prepare_events(day, ref)

        self.assertEqual(
            pitching.loc[pitching["s_no_key"] == 1, "event_datetime"].iloc[0].isoformat(),
            "2023-01-01T18:15:00+09:00",
        )
        self.assertEqual(
            pitching.loc[pitching["s_no_key"] == 2, "event_datetime"].iloc[0].isoformat(),
            "2023-04-02T00:00:00+09:00",
        )
        self.assertEqual(
            pitching.loc[pitching["s_no_key"] == 3, "event_datetime"].iloc[0].isoformat(),
            "2023-04-03T00:00:00+09:00",
        )


if __name__ == "__main__":
    unittest.main()
