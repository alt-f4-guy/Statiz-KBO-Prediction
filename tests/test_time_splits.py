import unittest

import pandas as pd


class TimeSplitTests(unittest.TestCase):
    def test_training_always_precedes_validation_and_same_day_stays_together(self):
        # 미래 경기 학습과 같은 날짜의 분할 경계 교차를 막는다.
        from time_splits import make_temporal_split_manifest

        dates = pd.to_datetime(
            [
                "2023-04-01T14:00:00+09:00",
                "2024-09-01T14:00:00+09:00",
                "2025-04-01T14:00:00+09:00",
                "2025-04-01T18:30:00+09:00",
                "2025-06-01T14:00:00+09:00",
                "2025-08-01T14:00:00+09:00",
                "2025-09-01T14:00:00+09:00",
                "2026-04-01T14:00:00+09:00",
            ]
        )
        data = pd.DataFrame(
            {"s_no": range(1, 9), "game_datetime": dates, "year": dates.year}
        )

        manifest = make_temporal_split_manifest(
            data, development_folds=2, calibration_fraction=0.25
        )

        by_id = data.set_index("s_no")["game_datetime"]
        for fold in manifest["development_folds"]:
            train_max = by_id.loc[fold["train_s_nos"]].max()
            validation_min = by_id.loc[fold["validation_s_nos"]].min()
            self.assertLess(train_max, validation_min)

        all_roles = {}
        for role in ("development_s_nos", "calibration_s_nos", "final_test_s_nos"):
            for s_no in manifest[role]:
                all_roles[s_no] = role
        self.assertEqual(all_roles[3], all_roles[4])

    def test_2026_is_excluded_from_development_and_calibration(self):
        # 최종 평가 시즌이 모델 선택에 들어가는 회귀를 막는다.
        from time_splits import make_temporal_split_manifest

        dates = pd.to_datetime(
            [
                "2023-04-01T14:00:00+09:00",
                "2024-04-01T14:00:00+09:00",
                "2025-04-01T14:00:00+09:00",
                "2025-08-01T14:00:00+09:00",
                "2025-09-01T14:00:00+09:00",
                "2026-04-01T14:00:00+09:00",
            ]
        )
        data = pd.DataFrame(
            {"s_no": range(1, 7), "game_datetime": dates, "year": dates.year}
        )

        manifest = make_temporal_split_manifest(data)

        selected = set(manifest["development_s_nos"]) | set(
            manifest["calibration_s_nos"]
        )
        self.assertNotIn(6, selected)
        self.assertEqual(manifest["final_test_s_nos"], [6])

    def test_tuning_frames_never_include_2026(self):
        # Optuna 목적함수에 최종 평가 시즌이 들어가는 회귀를 막는다.
        from tune_hyperparameters import build_tuning_folds

        dates = pd.to_datetime(
            [
                "2023-04-01T14:00:00+09:00",
                "2024-04-01T14:00:00+09:00",
                "2025-04-01T14:00:00+09:00",
                "2025-06-01T14:00:00+09:00",
                "2025-08-01T14:00:00+09:00",
                "2025-09-01T14:00:00+09:00",
                "2026-04-01T14:00:00+09:00",
            ]
        )
        data = pd.DataFrame(
            {
                "s_no": range(1, 8),
                "game_datetime": dates,
                "year": dates.year,
                "homeScore": [3, 2, 4, 3, 2, 1, 5],
                "awayScore": [2, 1, 2, 1, 1, 2, 4],
                "homeTeam": [1] * 7,
                "awayTeam": [2] * 7,
                "feature": range(7),
            }
        )

        folds, _ = build_tuning_folds(data)

        used_ids = {
            int(s_no)
            for train, validation in folds
            for s_no in pd.concat([train["s_no"], validation["s_no"]])
        }
        self.assertNotIn(7, used_ids)


if __name__ == "__main__":
    unittest.main()
