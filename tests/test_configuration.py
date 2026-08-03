import os
import unittest
from pathlib import Path
from unittest.mock import patch


class ConfigurationTests(unittest.TestCase):
    def test_extract_players_flattens_nested_lineup_payload(self):
        # API 응답 구조와 무관하게 선수 행만 공통으로 추출해야 한다.
        from statiz_api import extract_players

        players = extract_players(
            {
                "result_cd": 100,
                "lineup": [
                    {"p_no": 1, "name": "선수 A"},
                    {"bench": [{"p_no": 2, "name": "선수 B"}]},
                ],
            }
        )

        self.assertEqual([player["p_no"] for player in players], [1, 2])

    def test_missing_api_credentials_fail_before_client_creation(self):
        # 인증값이 하나라도 없으면 네트워크 클라이언트를 만들 수 없어야 한다.
        from pipeline_config import ConfigurationError, load_api_credentials

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ConfigurationError, "STATIZ_API_KEY"):
                load_api_credentials()

    def test_project_paths_do_not_depend_on_current_working_directory(self):
        # 실행 위치가 바뀌어도 데이터 경로는 프로젝트 루트 아래로 고정되어야 한다.
        from pipeline_config import PROJECT_ROOT, RAW_DATA_DIR

        original_cwd = Path.cwd()
        try:
            os.chdir("/tmp")
            self.assertEqual(RAW_DATA_DIR, PROJECT_ROOT / "data" / "raw")
        finally:
            os.chdir(original_cwd)

    def test_http_client_uses_timeout_and_reports_http_failure(self):
        # 무제한 대기나 조용한 None 반환으로 API 장애가 감춰지면 안 된다.
        from statiz_api import StatizAPI, StatizAPIError

        class Response:
            status_code = 500
            text = "server error"

            @staticmethod
            def json():
                return {}

        class Session:
            def __init__(self):
                self.timeout = None

            def get(self, *args, **kwargs):
                self.timeout = kwargs["timeout"]
                return Response()

        session = Session()
        client = StatizAPI("key", "secret", timeout=(3.05, 30), session=session)

        with self.assertRaisesRegex(StatizAPIError, "HTTP 500"):
            client.get("prediction/gameSchedule", {"year": 2026, "month": "07"})

        self.assertEqual(session.timeout, (3.05, 30))

    def test_http_429_uses_server_cooldown_before_retry(self):
        # 서버가 지정한 쿨다운보다 일찍 재요청하면 제한 상태가 반복된다.
        from statiz_api import StatizAPI

        class Response:
            def __init__(self, status_code, payload):
                self.status_code = status_code
                self.payload = payload
                self.text = ""

            def json(self):
                return self.payload

        class Session:
            def __init__(self):
                self.responses = [
                    Response(
                        429,
                        {
                            "rate_limit": {
                                "type": "burst",
                                "cooldown_sec": 53,
                            },
                            "result_cd": 429,
                        },
                    ),
                    Response(200, {"result_cd": 100}),
                ]

            def get(self, *args, **kwargs):
                return self.responses.pop(0)

        waits = []
        client = StatizAPI(
            "key",
            "secret",
            max_retries=1,
            session=Session(),
            sleep=waits.append,
        )

        result = client.get(
            "prediction/gameSchedule",
            {"year": 2026, "month": "07"},
        )

        self.assertEqual(result, {"result_cd": 100})
        self.assertEqual(waits, [53.0])

    def test_http_429_caps_server_cooldown_at_five_minutes(self):
        # 비정상적으로 큰 서버 값이 프로세스를 무기한 멈추면 안 된다.
        from statiz_api import StatizAPI, StatizAPIError

        class Response:
            status_code = 429
            text = ""

            @staticmethod
            def json():
                return {
                    "rate_limit": {"cooldown_sec": 600},
                    "result_cd": 429,
                }

        class Session:
            def get(self, *args, **kwargs):
                return Response()

        waits = []
        client = StatizAPI(
            "key",
            "secret",
            max_retries=1,
            session=Session(),
            sleep=waits.append,
        )

        with self.assertRaisesRegex(StatizAPIError, "HTTP 429"):
            client.get(
                "prediction/gameSchedule",
                {"year": 2026, "month": "07"},
            )

        self.assertEqual(waits, [300.0])


if __name__ == "__main__":
    unittest.main()
