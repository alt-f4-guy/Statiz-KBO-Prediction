import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from scripts.collect import collect_lineups, collect_rosters


class CollectionIntervalTests(unittest.TestCase):
    def test_lineup_requests_are_spaced_by_three_seconds(self):
        # 완료 경기 라인업 API 요청 사이에는 3초 간격을 둔다.
        class SuccessfulAPI:
            def get(self, path, params):
                return [{"p_no": 101}]

        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            pd.DataFrame(
                {
                    "s_no": [1],
                    "homeScore": [3],
                    "awayScore": [2],
                }
            ).to_csv(raw_dir / "games_master.csv", index=False)
            waits = []

            with (
                patch.object(collect_lineups, "RAW_DATA_DIR", raw_dir),
                patch.object(
                    collect_lineups,
                    "load_api_credentials",
                    return_value=SimpleNamespace(
                        api_key="key",
                        secret="secret",
                    ),
                ),
                patch.object(
                    collect_lineups,
                    "StatizAPI",
                    return_value=SuccessfulAPI(),
                ),
                patch.object(
                    collect_lineups.time,
                    "sleep",
                    side_effect=waits.append,
                ),
            ):
                collect_lineups.run_collection()

        self.assertEqual(waits, [3.0])

    def test_roster_requests_are_spaced_by_three_seconds(self):
        # 경기일·팀 로스터 API 요청 사이에는 3초 간격을 둔다.
        class SuccessfulAPI:
            def get(self, path, params):
                return {"101": {"p_no": 101}}

        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            pd.DataFrame(
                {
                    "year": [2026],
                    "month": [7],
                    "day": [30],
                    "homeTeam": [1],
                    "awayTeam": [2],
                }
            ).to_csv(raw_dir / "games_master.csv", index=False)
            waits = []

            with (
                patch.object(collect_rosters, "RAW_DATA_DIR", raw_dir),
                patch.object(
                    collect_rosters,
                    "load_api_credentials",
                    return_value=SimpleNamespace(
                        api_key="key",
                        secret="secret",
                    ),
                ),
                patch.object(
                    collect_rosters,
                    "StatizAPI",
                    return_value=SuccessfulAPI(),
                ),
                patch.object(
                    collect_rosters.time,
                    "sleep",
                    side_effect=waits.append,
                ),
            ):
                collect_rosters.run_roster_collection()

        self.assertEqual(waits, [3.0, 3.0])


if __name__ == "__main__":
    unittest.main()
