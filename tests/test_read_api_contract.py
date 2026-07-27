import unittest

import pandas as pd


class ReadApiContractTests(unittest.TestCase):
    def test_player_day_contract_uses_numeric_game_keys_and_record_fields(self):
        # 실제 응답은 s_no 열 대신 숫자 경기 ID를 최상위 키로 사용한다.
        from scripts.verify.read_api_contract import ENDPOINT_REQUIRED_FIELDS

        required = ENDPOINT_REQUIRED_FIELDS["prediction/playerDay"]

        self.assertNotIn("s_no", required)
        self.assertEqual(
            required,
            {"p_no", "gameDate", "homeScore", "awayScore"},
        )

    def test_summary_contains_no_response_body_or_credentials(self):
        # 검증 증거에 원문 응답과 인증 헤더가 남지 않아야 한다.
        from scripts.verify.read_api_contract import summarize_response

        summary = summarize_response(
            "prediction/gameLineup",
            {
                "result_cd": 100,
                "list": [
                    {
                        "p_no": 1,
                        "t_code": 1001,
                        "position": 1,
                        "battingOrder": "P",
                    }
                ],
            },
            {"p_no", "t_code", "position", "battingOrder"},
        )

        self.assertNotIn("response", summary)
        self.assertNotIn("headers", summary)
        self.assertNotIn("api_key", summary)
        self.assertTrue(summary["required_fields_present"])
        self.assertEqual(summary["row_count"], 1)
        self.assertEqual(summary["http_status"], 200)

    def test_smoke_targets_use_completed_game_pitcher_batter_and_roster(self):
        # 쓰기 없이 재현 가능한 읽기 표본을 로컬 원천 데이터에서 고정한다.
        from scripts.verify.read_api_contract import select_smoke_targets

        games = pd.DataFrame(
            {
                "s_no": [1, 2],
                "year": [2026, 2026],
                "month": [7, 7],
                "day": [27, 28],
                "homeTeam": [1001, 2002],
                "homeScore": [3.0, pd.NA],
                "awayScore": [2.0, pd.NA],
            }
        )
        lineups = pd.DataFrame(
            {
                "s_no": [1, 1],
                "p_no": [10, 20],
                "position": [1, 3],
                "battingOrder": ["P", "1"],
                "starting": ["Y", "Y"],
            }
        )
        rosters = pd.DataFrame(
            {
                "p_no": [10],
                "t_code": [1001],
                "pj_date": ["2026-07-27"],
            }
        )

        targets = select_smoke_targets(games, lineups, rosters)

        self.assertEqual(targets["s_no"], 1)
        self.assertEqual(targets["pitcher_p_no"], 10)
        self.assertEqual(targets["batter_p_no"], 20)
        self.assertEqual(targets["roster_team"], 1001)
        self.assertEqual(targets["roster_date"], "2026-07-27")


if __name__ == "__main__":
    unittest.main()
