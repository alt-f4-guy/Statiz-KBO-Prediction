import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd


class PlayerStatsCollectionTests(unittest.TestCase):
    def test_game_day_population_uses_latest_roster_for_today_teams(self):
        from player_stats_collection import load_game_day_player_population

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games = root / "games.csv"
            rosters = root / "rosters.csv"
            pd.DataFrame(
                {
                    "year": [2026, 2026, 2026],
                    "month": [7, 7, 7],
                    "day": [28, 28, 27],
                    "state": [1, 4, 3],
                    "homeTeam": [1001, 3001, 5002],
                    "awayTeam": [2002, 4003, 6002],
                }
            ).to_csv(games, index=False)
            pd.DataFrame(
                {
                    "pj_date": [
                        "2026-07-27",
                        "2026-07-28",
                        "2026-07-28",
                        "2026-07-29",
                        "2026-07-27",
                    ],
                    "t_code": [1001, 1001, 1001, 1001, 2002],
                    "p_no": [10, 11, 13, 12, 20],
                }
            ).to_csv(rosters, index=False)

            result = load_game_day_player_population(
                rosters,
                games,
                date(2026, 7, 28),
            )

        self.assertEqual(result, [11, 13, 20])

    def test_game_day_population_is_empty_without_games(self):
        from player_stats_collection import load_game_day_player_population

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games = root / "games.csv"
            rosters = root / "rosters.csv"
            pd.DataFrame(
                {
                    "year": [2026],
                    "month": [7],
                    "day": [27],
                    "state": [3],
                    "homeTeam": [1001],
                    "awayTeam": [2002],
                }
            ).to_csv(games, index=False)
            pd.DataFrame(
                {
                    "pj_date": ["2026-07-27"],
                    "t_code": [1001],
                    "p_no": [10],
                }
            ).to_csv(rosters, index=False)

            result = load_game_day_player_population(
                rosters,
                games,
                date(2026, 7, 28),
            )

        self.assertEqual(result, [])

    def test_game_day_population_rejects_missing_team_roster(self):
        from player_stats_collection import load_game_day_player_population

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games = root / "games.csv"
            rosters = root / "rosters.csv"
            pd.DataFrame(
                {
                    "year": [2026],
                    "month": [7],
                    "day": [28],
                    "state": [1],
                    "homeTeam": [1001],
                    "awayTeam": [2002],
                }
            ).to_csv(games, index=False)
            pd.DataFrame(
                {
                    "pj_date": ["2026-07-28"],
                    "t_code": [1001],
                    "p_no": [10],
                }
            ).to_csv(rosters, index=False)

            with self.assertRaisesRegex(ValueError, "2002"):
                load_game_day_player_population(
                    rosters,
                    games,
                    date(2026, 7, 28),
                )

    def test_reusable_years_include_past_and_today_current_success(self):
        from player_stats_collection import reusable_player_years

        snapshots = pd.DataFrame(
            {
                "p_no": [10, 10, 20, 30],
                "year_req": [2025, 2026, 2026, 2025],
                "fetched_at": [
                    "2025-10-01T12:00:00+09:00",
                    "2026-07-28T08:00:00+09:00",
                    "2026-07-27T23:50:00+09:00",
                    "2026-07-28T08:00:00+09:00",
                ],
                "response_status": [
                    "success",
                    "success",
                    "success",
                    "error",
                ],
            }
        )

        result = reusable_player_years(
            snapshots,
            current_year=2026,
            target_date=date(2026, 7, 28),
        )

        self.assertEqual(result, {(10, 2025), (10, 2026)})

    def test_today_current_success_is_not_requested_again(self):
        from player_stats_collection import years_to_collect

        completed = {(10, 2025), (10, 2026)}
        result = years_to_collect(
            p_no=10,
            years=[2025, 2026],
            completed=completed,
        )

        self.assertEqual(result, [])

    def test_stale_current_success_is_requested_again(self):
        from player_stats_collection import years_to_collect

        result = years_to_collect(
            p_no=20,
            years=[2025, 2026],
            completed={(20, 2025)},
        )

        self.assertEqual(result, [2026])

    def test_player_requests_are_spaced_by_configured_interval(self):
        from player_stats_collection import collect_player_snapshots

        class SuccessfulAPI:
            def get(self, path, params):
                if path == "prediction/playerSeason":
                    return {
                        "result_cd": 100,
                        "batting": {
                            "list": [
                                {"year": 2025},
                                {"year": 2026},
                            ]
                        },
                    }
                return {"result_cd": 100, "games": []}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            waits = []

            failures = collect_player_snapshots(
                SuccessfulAPI(),
                [10],
                [2025, 2026],
                2026,
                root / "day.csv",
                root / "season.csv",
                target_date=date(2026, 7, 28),
                request_interval=0.3,
                sleep=waits.append,
            )

        self.assertEqual(failures, [])
        self.assertEqual(waits, [0.3, 0.3, 0.3])

    def test_player_collection_reports_progress_after_each_player(self):
        from player_stats_collection import collect_player_snapshots

        class SuccessfulAPI:
            def get(self, path, params):
                if path == "prediction/playerSeason":
                    return {
                        "result_cd": 100,
                        "batting": {"list": [{"year": 2025}]},
                    }
                return {"result_cd": 100, "games": []}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress_events = []

            failures = collect_player_snapshots(
                SuccessfulAPI(),
                [10, 20],
                [2025],
                2026,
                root / "day.csv",
                root / "season.csv",
                target_date=date(2026, 7, 28),
                sleep=lambda _: None,
                progress_callback=lambda *event: progress_events.append(event),
            )

        self.assertEqual(failures, [])
        self.assertEqual(
            progress_events,
            [(1, 10, 2, 0), (2, 20, 4, 0)],
        )

    def test_final_api_failure_stops_before_next_player(self):
        from player_stats_collection import collect_player_snapshots
        from statiz_api import StatizAPIError

        class FailingAPI:
            def __init__(self):
                self.calls = []

            def get(self, path, params):
                self.calls.append((path, int(params["p_no"])))
                raise StatizAPIError("limited")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            api = FailingAPI()

            with self.assertRaisesRegex(StatizAPIError, "limited"):
                collect_player_snapshots(
                    api,
                    [10, 20],
                    [2025, 2026],
                    2026,
                    root / "day.csv",
                    root / "season.csv",
                    target_date=date(2026, 7, 28),
                    request_interval=0.3,
                    sleep=lambda _: None,
                )

        self.assertEqual(
            api.calls,
            [("prediction/playerSeason", 10)],
        )


if __name__ == "__main__":
    unittest.main()
