import unittest


class RealtimePredictionTests(unittest.TestCase):
    def test_api_percent_is_always_home_probability(self):
        # 원정팀 선택 시 percent를 선택 팀 확률로 뒤집는 회귀를 막는다.
        from realtime_prediction import build_prediction_payload

        payload = build_prediction_payload(
            ptt_idx="05",
            s_no=1,
            home_team_name="홈",
            away_team_name="원정",
            home_win_probability=0.40,
            update_time="2026-07-26 12:00:00",
        )

        self.assertEqual(payload["predictWinTeam"], "원정")
        self.assertEqual(payload["percent"], 40.0)
        self.assertEqual(payload["selected_team_probability"], 60.0)


if __name__ == "__main__":
    unittest.main()
