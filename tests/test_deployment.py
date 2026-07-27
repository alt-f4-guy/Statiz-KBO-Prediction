import tempfile
import unittest
from pathlib import Path

import pandas as pd


class DeploymentTests(unittest.TestCase):
    def test_deployment_context_is_reproducible(self):
        # 같은 Git 커밋과 파일이면 배포 식별자와 체크섬이 같아야 한다.
        from deployment import build_deployment_context

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "model_path": "artifacts/models/best_model.joblib",
                "data_path": "data/final/final_training_set_v9.csv",
                "split_manifest": "data/final/time_split_manifest.json",
            }
            for index, relative_path in enumerate(paths.values()):
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"artifact-{index}".encode())
            metadata = {
                **paths,
                "prospective_start_date": "2026-07-28",
                "baseline_home_probability": 0.52,
            }

            first = build_deployment_context(root, metadata, "abc123")
            second = build_deployment_context(root, metadata, "abc123")

            self.assertEqual(first, second)
            self.assertEqual(len(first["deployment_id"]), 16)
            self.assertEqual(first["git_commit"], "abc123")
            self.assertEqual(first["prospective_start_date"], "2026-07-28")

    def test_deployment_context_rejects_absolute_artifact_paths(self):
        # 다른 작업 폴더의 절대 경로가 메타데이터에 고정되는 회귀를 막는다.
        from deployment import build_deployment_context

        metadata = {
            "model_path": "/tmp/model.joblib",
            "data_path": "data/final/final_training_set_v9.csv",
            "split_manifest": "data/final/time_split_manifest.json",
            "prospective_start_date": "2026-07-28",
            "baseline_home_probability": 0.52,
        }

        with self.assertRaisesRegex(ValueError, "상대 경로"):
            build_deployment_context(Path("/tmp/project"), metadata, "abc123")

    def test_evaluation_role_starts_on_registered_seoul_date(self):
        # 전향적 시작일 전 예측은 공학적 회귀 검사로 분리해야 한다.
        from deployment import evaluation_role_for

        before = pd.Timestamp("2026-07-27T23:59:59+09:00")
        start = pd.Timestamp("2026-07-28T00:00:00+09:00")

        self.assertEqual(
            evaluation_role_for(before, "2026-07-28"),
            "engineering_regression",
        )
        self.assertEqual(
            evaluation_role_for(start, "2026-07-28"),
            "prospective_holdout",
        )


if __name__ == "__main__":
    unittest.main()
