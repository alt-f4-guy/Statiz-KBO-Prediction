import unittest

import pandas as pd


class RealtimePredictionTests(unittest.TestCase):
    def test_game_progress_uses_matchup_and_seoul_start_time(self):
        # 경기 행은 원정 @ 홈과 서울 현지 시작 시각을 표시한다.
        from predict_2026 import _game_progress

        game = {
            "s_no": 20260496,
            "homeTeam": 5002,
            "awayTeam": 10001,
            "homeTeamName": "LG",
            "awayTeamName": "키움",
        }
        start = pd.Timestamp("2026-07-28T18:30:00+09:00")

        progress = _game_progress(game, start)

        self.assertEqual(progress.s_no, 20260496)
        self.assertEqual(progress.matchup, "키움 @ LG")
        self.assertEqual(progress.start_time, "18:30")

    def test_terminal_games_include_success_and_offline_only(self):
        # 만료 이력은 재처리하고 성공 제출·오프라인 예측만 완료로 복원한다.
        from predict_2026 import _terminal_game_ids

        history = pd.DataFrame(
            {
                "record_type": [
                    "delivery",
                    "delivery",
                    "expired",
                    "offline_prediction",
                ],
                "api_status": [
                    "success",
                    "failed",
                    "expired",
                    "not_submitted",
                ],
                "s_no": [1, 2, 3, 4],
            }
        )

        self.assertEqual(_terminal_game_ids(history), {1, 4})

    def test_offline_record_preserves_prediction_and_marks_non_submission(self):
        # 재사용한 확률·모델은 유지하고 경기 후 기록은 회고 진단으로 격리한다.
        from realtime_prediction import build_offline_prediction_record

        prediction = {
            "recorded_at": "2026-07-28T17:00:00+09:00",
            "record_type": "prediction",
            "s_no": 1,
            "home_win_probability": 0.61,
            "model_type": "primary",
            "evaluation_role": "prospective_holdout",
            "api_status": "pending",
            "error_type": "old",
        }

        result = build_offline_prediction_record(
            prediction,
            recorded_at="2026-07-28T19:00:00+09:00",
        )

        self.assertEqual(
            result["recorded_at"],
            "2026-07-28T19:00:00+09:00",
        )
        self.assertEqual(result["record_type"], "offline_prediction")
        self.assertEqual(result["api_status"], "not_submitted")
        self.assertEqual(
            result["evaluation_role"],
            "retrospective_diagnostic",
        )
        self.assertEqual(result["prediction_mode"], "offline_after_start")
        self.assertEqual(result["home_win_probability"], 0.61)
        self.assertEqual(result["model_type"], "primary")
        self.assertEqual(result["error_type"], "")

    def test_prediction_window_closes_at_game_start(self):
        # 경기 시작 시각과 그 이후의 신규 예측·전송을 차단한다.
        from realtime_prediction import prediction_window_is_open

        start = pd.Timestamp("2026-07-28T18:30:00+09:00")

        self.assertTrue(
            prediction_window_is_open(start - pd.Timedelta("1ns"), start)
        )
        self.assertFalse(prediction_window_is_open(start, start))
        self.assertFalse(
            prediction_window_is_open(start + pd.Timedelta("1s"), start)
        )

    def test_prediction_window_requires_timezone_aware_values(self):
        # 로컬 시각을 UTC로 오인하여 경기 후 예측하는 오류를 막는다.
        from realtime_prediction import prediction_window_is_open

        aware = pd.Timestamp("2026-07-28T18:30:00+09:00")

        with self.assertRaisesRegex(ValueError, "시간대"):
            prediction_window_is_open(aware.tz_localize(None), aware)

    def test_feature_prior_usage_rate_uses_source_columns_only(self):
        # 수치 열은 제외하고 실제 피처 출처 열의 리그 사전분포 비율만 센다.
        from realtime_prediction import feature_prior_usage_rate

        row = pd.Series(
            {
                "home_sp_source": "league_prior",
                "away_sp_source": "current_season",
                "home_sp_fip": 4.0,
            }
        )

        self.assertEqual(feature_prior_usage_rate(row), 0.5)
        self.assertTrue(pd.isna(feature_prior_usage_rate(None)))

    def test_failed_delivery_record_preserves_prediction_audit_context(self):
        # 최종 전송 실패도 원래 예측과 배포 버전을 잃지 않고 추가 기록한다.
        from realtime_prediction import build_delivery_record

        prediction = {
            "s_no": 1,
            "game_datetime": "2026-07-28T18:30:00+09:00",
            "deployment_id": "deploy-a",
            "home_win_probability": 0.4,
            "api_status": "pending",
        }

        result = build_delivery_record(
            prediction,
            recorded_at="2026-07-28T17:00:00+09:00",
            api_status="failed",
            error_type="StatizAPIError",
        )

        self.assertEqual(result["record_type"], "delivery")
        self.assertEqual(result["api_status"], "failed")
        self.assertEqual(result["deployment_id"], "deploy-a")
        self.assertEqual(result["home_win_probability"], 0.4)
        self.assertEqual(result["error_type"], "StatizAPIError")

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
