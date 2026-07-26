import os
import unittest
from pathlib import Path
from unittest.mock import patch


class ConfigurationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
