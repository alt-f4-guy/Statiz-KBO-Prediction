import tempfile
import unittest
from pathlib import Path

import pandas as pd


class PlayerStatsCollectionTests(unittest.TestCase):
    def test_roster_only_pitcher_is_included_in_player_population(self):
        # 모집단을 라인업으로 되돌리면 로스터에만 있는 구원 투수를 놓쳐야 한다.
        from player_stats_collection import load_player_population

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lineups = root / "lineups.csv"
            rosters = root / "rosters.csv"
            pd.DataFrame({"p_no": [10, 20]}).to_csv(lineups, index=False)
            pd.DataFrame({"p_no": [20, 30]}).to_csv(rosters, index=False)

            result = load_player_population(lineups, rosters)

        self.assertEqual(result, [10, 20, 30])

    def test_current_year_is_always_requested_but_completed_past_year_is_reused(self):
        # 완료 상태만 보고 현재 시즌 호출을 생략하는 회귀를 막는다.
        from player_stats_collection import years_to_collect

        completed = {(10, 2024), (10, 2026)}
        result = years_to_collect(
            p_no=10,
            years=[2024, 2025, 2026],
            current_year=2026,
            completed=completed,
        )

        self.assertEqual(result, [2025, 2026])

    def test_failed_snapshot_does_not_count_as_completed(self):
        # 실패 행이 존재한다는 이유로 과거 시즌 재시도를 막으면 안 된다.
        from player_stats_collection import completed_player_years

        snapshots = pd.DataFrame(
            {
                "p_no": [10, 10, 20],
                "year_req": [2024, 2025, 2024],
                "response_status": ["success", "error", "success"],
            }
        )

        self.assertEqual(
            completed_player_years(snapshots),
            {(10, 2024), (20, 2024)},
        )


if __name__ == "__main__":
    unittest.main()
