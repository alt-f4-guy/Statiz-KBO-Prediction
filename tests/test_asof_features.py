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


if __name__ == "__main__":
    unittest.main()
