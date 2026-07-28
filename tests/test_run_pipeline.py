import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_pipeline


class RunPipelineTests(unittest.TestCase):
    def test_daily_phases_only_collect_process_and_predict(self):
        modules = [module for module, _ in run_pipeline.DAILY_PHASES]

        self.assertEqual(
            modules,
            [
                "scripts.collect.collect_schedule",
                "scripts.collect.collect_lineups",
                "scripts.collect.collect_rosters",
                "scripts.collect.collect_player_stats",
                "scripts.build.process_raw_data",
                "scripts.ops.predict_2026",
            ],
        )

    def test_prediction_runs_after_player_collection_and_processing(self):
        # 최신 선수 스냅샷을 정형화한 뒤에만 예측을 시작한다.
        modules = [module for module, _ in run_pipeline.DAILY_PHASES]

        self.assertGreater(
            modules.index("scripts.ops.predict_2026"),
            modules.index("scripts.collect.collect_player_stats"),
        )
        self.assertGreater(
            modules.index("scripts.ops.predict_2026"),
            modules.index("scripts.build.process_raw_data"),
        )

    def test_environment_file_loads_missing_values_without_overriding_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "STATIZ_API_KEY=file-key\n"
                "STATIZ_SECRET='file-secret'\n"
                'STATIZ_PTT_IDX="file-account"\n',
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"STATIZ_API_KEY": "shell-key"},
                clear=True,
            ):
                run_pipeline.load_runtime_environment(env_path)

                self.assertEqual(os.environ["STATIZ_API_KEY"], "shell-key")
                self.assertEqual(os.environ["STATIZ_SECRET"], "file-secret")
                self.assertEqual(os.environ["STATIZ_PTT_IDX"], "file-account")

    def test_missing_environment_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"

            with self.assertRaisesRegex(RuntimeError, "환경파일"):
                run_pipeline.load_runtime_environment(env_path)

    def test_missing_required_environment_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "STATIZ_API_KEY=file-key\nSTATIZ_SECRET=file-secret\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "STATIZ_PTT_IDX"):
                    run_pipeline.load_runtime_environment(env_path)

    def test_main_returns_failure_when_a_daily_phase_fails(self):
        with (
            patch.object(run_pipeline, "load_runtime_environment"),
            patch.object(
                run_pipeline,
                "run_script",
                return_value=(False, 0),
            ) as run_script,
            patch.object(run_pipeline, "console"),
        ):
            exit_code = run_pipeline.main()

        self.assertEqual(exit_code, 1)
        self.assertEqual(run_script.call_count, 1)

    def test_main_returns_success_after_prediction_phase_finishes(self):
        with (
            patch.object(run_pipeline, "load_runtime_environment"),
            patch.object(
                run_pipeline,
                "run_script",
                return_value=(True, 0.01),
            ) as run_script,
            patch.object(run_pipeline, "console"),
        ):
            exit_code = run_pipeline.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(run_script.call_count, len(run_pipeline.DAILY_PHASES))


if __name__ == "__main__":
    unittest.main()
