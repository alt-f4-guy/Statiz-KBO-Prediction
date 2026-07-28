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

    def test_local_started_games_selects_only_started_games_today(self):
        # 로컬 대체는 오늘 시작한 경기만 반환하고 미래·과거 경기는 제외한다.
        from predict_2026 import _local_started_games

        games = pd.DataFrame(
            {
                "s_no": [1, 2, 3],
                "gameDate": [
                    1785231000,  # 2026-07-28 18:30 KST
                    1785317400,  # 2026-07-29 18:30 KST
                    1785144600,  # 2026-07-27 18:30 KST
                ],
                "homeTeam": [5002, 5002, 5002],
                "awayTeam": [10001, 10001, 10001],
            }
        )

        result = _local_started_games(
            games,
            pd.Timestamp("2026-07-28T19:00:00+09:00"),
        )

        self.assertEqual([game["s_no"] for game in result], [1])

    def test_local_started_games_excludes_future_and_invalid_start_times(self):
        # 오늘 행이라도 시작 전이거나 시각 파싱이 불가능하면 사용하지 않는다.
        from predict_2026 import _local_started_games

        games = pd.DataFrame(
            {
                "s_no": [1, 2],
                "gameDate": [1785231000, "invalid"],
                "homeTeam": [5002, 5002],
                "awayTeam": [10001, 10001],
            }
        )

        result = _local_started_games(
            games,
            pd.Timestamp("2026-07-28T18:00:00+09:00"),
        )

        self.assertEqual(result, [])

    def test_schedule_failure_uses_started_local_games_without_waiting(self):
        # 일정 요청 실패는 외부 sleep 없이 이미 시작한 로컬 경기로 전환한다.
        from predict_2026 import _load_today_games
        from statiz_api import StatizAPIError

        class FailedAPI:
            def get(self, endpoint, params):
                raise StatizAPIError("HTTP 429")

        games = pd.DataFrame(
            {
                "s_no": [1],
                "gameDate": [1785231000],
                "homeTeam": [5002],
                "awayTeam": [10001],
            }
        )

        result, used_local = _load_today_games(
            FailedAPI(),
            games,
            now_utc=pd.Timestamp("2026-07-28T19:00:00+09:00"),
        )

        self.assertTrue(used_local)
        self.assertEqual([game["s_no"] for game in result], [1])

    def test_schedule_failure_without_started_local_games_reraises(self):
        # 경기 전에는 로컬 일정으로 제출하지 않고 일정 오류를 상위에 전달한다.
        from predict_2026 import _load_today_games
        from statiz_api import StatizAPIError

        class FailedAPI:
            def get(self, endpoint, params):
                raise StatizAPIError("HTTP 429")

        future = pd.DataFrame(
            {
                "s_no": [1],
                "gameDate": [1785231000],
                "homeTeam": [5002],
                "awayTeam": [10001],
            }
        )

        with self.assertRaisesRegex(StatizAPIError, "429"):
            _load_today_games(
                FailedAPI(),
                future,
                now_utc=pd.Timestamp("2026-07-28T18:00:00+09:00"),
            )

    def test_successful_schedule_keeps_api_games(self):
        # 정상 응답이 있으면 로컬 데이터가 아니라 API의 오늘 일정을 사용한다.
        from predict_2026 import _load_today_games

        class SuccessfulAPI:
            def get(self, endpoint, params):
                return {
                    "20260728": [
                        {
                            "s_no": 10,
                            "homeTeam": 5002,
                            "awayTeam": 10001,
                        }
                    ]
                }

        result, used_local = _load_today_games(
            SuccessfulAPI(),
            pd.DataFrame(),
            now_utc=pd.Timestamp("2026-07-28T17:00:00+09:00"),
        )

        self.assertFalse(used_local)
        self.assertEqual([game["s_no"] for game in result], [10])

    def test_empty_schedule_uses_started_local_games(self):
        # 빈 정상 응답도 오늘 경기가 시작했다면 로컬 오프라인 예측으로 전환한다.
        from predict_2026 import _load_today_games

        class EmptyAPI:
            def get(self, endpoint, params):
                return {}

        games = pd.DataFrame(
            {
                "s_no": [1],
                "gameDate": [1785231000],
                "homeTeam": [5002],
                "awayTeam": [10001],
            }
        )

        result, used_local = _load_today_games(
            EmptyAPI(),
            games,
            now_utc=pd.Timestamp("2026-07-28T19:00:00+09:00"),
        )

        self.assertTrue(used_local)
        self.assertEqual([game["s_no"] for game in result], [1])

    def test_realtime_system_disables_internal_read_retries(self):
        # 429가 300초 내부 sleep으로 들어가지 않도록 예측 인스턴스만 끈다.
        from types import SimpleNamespace
        from unittest.mock import patch

        from prediction_progress import PredictionProgressDisplay
        from predict_2026 import _run_realtime_prediction_system

        captured = {}

        def stop_after_constructor(*args, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("constructor observed")

        with (
            patch(
                "predict_2026.load_api_credentials",
                return_value=SimpleNamespace(
                    api_key="key",
                    secret="secret",
                    ptt_idx="05",
                ),
            ),
            patch(
                "predict_2026.StatizAPI",
                side_effect=stop_after_constructor,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "constructor observed",
            ):
                _run_realtime_prediction_system(
                    PredictionProgressDisplay()
                )

        self.assertEqual(captured["max_retries"], 0)

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

    def test_today_games_complete_requires_every_game_to_be_terminal(self):
        # 오늘 경기 전체가 완료된 경우에만 추가 폴링 없이 종료할 수 있다.
        from predict_2026 import _today_games_complete

        games = [{"s_no": 1}, {"s_no": 2}]

        self.assertTrue(_today_games_complete(games, {1, 2}))
        self.assertFalse(_today_games_complete(games, {1}))

    def test_completed_game_progress_restores_offline_result(self):
        # 재시작하면 오프라인 예측 로그의 모델·확률·미제출 상태를 복원한다.
        from predict_2026 import _completed_game_progress

        history = pd.DataFrame(
            [
                {
                    "record_type": "offline_prediction",
                    "api_status": "not_submitted",
                    "s_no": 1,
                    "model_type": "primary",
                    "home_win_probability": 0.61,
                }
            ]
        )
        game = {
            "s_no": 1,
            "homeTeam": 5002,
            "awayTeam": 10001,
            "homeTeamName": "LG",
            "awayTeamName": "키움",
        }

        progress = _completed_game_progress(
            game,
            pd.Timestamp("2026-07-28T18:30:00+09:00"),
            history,
        )

        self.assertEqual(progress.step, 6)
        self.assertEqual(progress.status, "미제출 예측 완료 · 홈 승률 61.0%")
        self.assertEqual(progress.model, "primary")
        self.assertEqual(progress.delivery, "미제출")

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

    def test_complete_prediction_after_start_skips_api_and_logs_offline(self):
        # 경기 시작 후에는 저장 API를 호출하지 않고 오프라인 로그만 남긴다.
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from predict_2026 import _complete_prediction

        class CountingAPI:
            def __init__(self):
                self.post_calls = []

            def post(self, endpoint, payload):
                self.post_calls.append((endpoint, payload))
                return {"result_cd": 100}

        class Display:
            def __init__(self):
                self.changes = []

            def advance(self, s_no, **changes):
                self.changes.append((s_no, changes))

        api = CountingAPI()
        display = Display()
        prediction = {
            "recorded_at": "2026-07-28T17:00:00+09:00",
            "record_type": "prediction",
            "s_no": 1,
            "game_datetime": "2026-07-28T18:30:00+09:00",
            "home_win_probability": 0.61,
            "model_type": "primary",
            "evaluation_role": "prospective_holdout",
            "api_status": "pending",
        }

        with TemporaryDirectory() as directory:
            log_path = Path(directory) / "prediction_log.csv"
            with patch("predict_2026.PREDICTION_LOG", log_path):
                record, terminal = _complete_prediction(
                    api,
                    display,
                    prediction,
                    {"s_no": 1},
                    game_time=pd.Timestamp(
                        "2026-07-28T18:30:00+09:00"
                    ),
                    now_utc=pd.Timestamp(
                        "2026-07-28T19:00:00+09:00"
                    ),
                )
            saved = pd.read_csv(log_path)

        self.assertEqual(api.post_calls, [])
        self.assertTrue(terminal)
        self.assertEqual(record["record_type"], "offline_prediction")
        self.assertEqual(record["api_status"], "not_submitted")
        self.assertEqual(saved["record_type"].tolist(), ["offline_prediction"])
        self.assertEqual(display.changes[-1][1]["delivery"], "미제출")

    def test_complete_prediction_before_start_keeps_successful_submission(self):
        # 경기 시작 전에는 기존 저장 API를 호출하고 성공 delivery를 남긴다.
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from predict_2026 import _complete_prediction

        class SuccessfulAPI:
            def __init__(self):
                self.post_calls = []

            def post(self, endpoint, payload):
                self.post_calls.append((endpoint, payload))
                return {"result_cd": 100}

        class Display:
            def advance(self, s_no, **changes):
                pass

        api = SuccessfulAPI()
        prediction = {
            "recorded_at": "2026-07-28T17:00:00+09:00",
            "record_type": "prediction",
            "s_no": 1,
            "game_datetime": "2026-07-28T18:30:00+09:00",
            "home_win_probability": 0.61,
            "model_type": "primary",
            "api_status": "pending",
        }

        with TemporaryDirectory() as directory:
            log_path = Path(directory) / "prediction_log.csv"
            with patch("predict_2026.PREDICTION_LOG", log_path):
                record, terminal = _complete_prediction(
                    api,
                    Display(),
                    prediction,
                    {
                        "ptt_idx": "05",
                        "s_no": 1,
                        "homeTeam": "홈",
                        "awayTeam": "원정",
                        "predictWinTeam": "홈",
                        "percent": 61.0,
                        "update_time": "2026-07-28 18:00:00",
                    },
                    game_time=pd.Timestamp(
                        "2026-07-28T18:30:00+09:00"
                    ),
                    now_utc=pd.Timestamp(
                        "2026-07-28T18:00:00+09:00"
                    ),
                )
            saved = pd.read_csv(log_path)

        self.assertEqual(
            [call[0] for call in api.post_calls],
            ["prediction/savePrediction"],
        )
        self.assertTrue(terminal)
        self.assertEqual(record["record_type"], "delivery")
        self.assertEqual(record["api_status"], "success")
        self.assertEqual(saved["record_type"].tolist(), ["delivery"])

    def test_incomplete_lineup_after_start_does_not_wait(self):
        # 경기 후에는 라인업을 기다리지 않고 대체 확률 계산으로 진행한다.
        from predict_2026 import _lineup_wait_required

        start = pd.Timestamp("2026-07-28T18:30:00+09:00")

        self.assertFalse(
            _lineup_wait_required(
                submit_before_start=False,
                complete=False,
                now_utc=start + pd.Timedelta(minutes=30),
                deadline=start - pd.Timedelta(minutes=30),
            )
        )

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
