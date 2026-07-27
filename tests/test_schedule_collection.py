import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from scripts.collect import collect_schedule
from statiz_api import StatizAPIError


class ScheduleCollectionTests(unittest.TestCase):
    def test_complete_history_requests_only_current_month(self):
        existing = pd.DataFrame(
            {
                "year": [2023, 2026],
                "month": [4, 6],
                "state": [3, 3],
                "homeScore": [3, 5],
                "awayScore": [2, 4],
            }
        )

        months = collect_schedule.schedule_months_to_fetch(
            existing,
            datetime(2026, 7, 27),
        )

        self.assertEqual(months, [("2026", "07")])

    def test_unfinished_past_month_is_requested_with_current_month(self):
        existing = pd.DataFrame(
            {
                "year": [2023, 2025, 2026, 2026],
                "month": [4, 5, 6, 8],
                "state": [1, 4, 3, 1],
                "homeScore": [pd.NA, pd.NA, 5, pd.NA],
                "awayScore": [2, pd.NA, 4, pd.NA],
            }
        )

        months = collect_schedule.schedule_months_to_fetch(
            existing,
            datetime(2026, 7, 27),
        )

        self.assertEqual(
            months,
            [("2023", "04"), ("2026", "07")],
        )

    def test_empty_history_requests_months_from_2023_to_current_month(self):
        months = collect_schedule.schedule_months_to_fetch(
            pd.DataFrame(),
            datetime(2023, 3, 15),
        )

        self.assertEqual(
            months,
            [
                ("2023", "01"),
                ("2023", "02"),
                ("2023", "03"),
            ],
        )

    def test_month_failure_stops_collection(self):
        class FailingAPI:
            def get(self, path, params):
                raise StatizAPIError("요청 제한")

        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            pd.DataFrame(
                {
                    "year": [2026],
                    "month": [7],
                    "state": [3],
                    "homeScore": [3],
                    "awayScore": [2],
                }
            ).to_csv(raw_dir / "games_master.csv", index=False)

            with (
                patch.object(collect_schedule, "RAW_DATA_DIR", raw_dir),
                patch.object(
                    collect_schedule,
                    "load_api_credentials",
                    return_value=SimpleNamespace(
                        api_key="key",
                        secret="secret",
                    ),
                ),
                patch.object(
                    collect_schedule,
                    "StatizAPI",
                    return_value=FailingAPI(),
                ),
                patch.object(collect_schedule.time, "sleep"),
            ):
                with self.assertRaisesRegex(StatizAPIError, "요청 제한"):
                    collect_schedule.run_schedule_collection()


if __name__ == "__main__":
    unittest.main()
