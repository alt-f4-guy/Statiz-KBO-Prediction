import unittest


class PredictionProgressTests(unittest.TestCase):
    def test_new_game_starts_at_schedule_check(self):
        # 새 경기는 일정 확인 단계와 제출 대기 상태로 시작한다.
        from prediction_progress import create_game_progress

        progress = create_game_progress(
            s_no=20260496,
            matchup="키움 @ LG",
            start_time="18:30",
        )

        self.assertEqual(progress.step, 1)
        self.assertEqual(progress.status, "경기 확인")
        self.assertEqual(progress.model, "-")
        self.assertEqual(progress.delivery, "대기")

    def test_progress_never_moves_backward(self):
        # 늦게 도착한 폴링 상태가 이미 진행된 단계를 되돌리면 안 된다.
        from prediction_progress import (
            advance_game_progress,
            create_game_progress,
        )

        progress = create_game_progress(1, "원정 @ 홈", "18:30")
        progressed = advance_game_progress(
            progress,
            step=4,
            status="모델 추론",
            model="primary",
        )
        stale = advance_game_progress(
            progressed,
            step=2,
            status="라인업 대기",
        )

        self.assertEqual(stale, progressed)

    def test_summary_counts_success_expired_waiting_and_failure(self):
        # 성공·만료는 완료, 실패는 실패, 나머지는 대기로 집계한다.
        from prediction_progress import (
            advance_game_progress,
            create_game_progress,
            summarize_progress,
        )

        waiting = create_game_progress(1, "A @ B", "18:30")
        success = advance_game_progress(
            create_game_progress(2, "C @ D", "18:30"),
            step=6,
            status="제출 완료",
            delivery="성공",
        )
        expired = advance_game_progress(
            create_game_progress(3, "E @ F", "18:30"),
            step=6,
            status="경기 시작",
            delivery="만료",
        )
        failed = advance_game_progress(
            create_game_progress(4, "G @ H", "18:30"),
            step=6,
            status="제출 실패",
            delivery="실패",
            error_type="StatizAPIError",
        )

        summary = summarize_progress(
            {1: waiting, 2: success, 3: expired, 4: failed}
        )

        self.assertEqual(summary.total, 4)
        self.assertEqual(summary.completed, 2)
        self.assertEqual(summary.waiting, 1)
        self.assertEqual(summary.failed, 1)

    def test_rendered_view_contains_preparation_summary_and_game_rows(self):
        # 실제 Rich 출력에 준비 상태, 전체 집계와 경기별 상태가 나타난다.
        from rich.console import Console

        from prediction_progress import (
            advance_game_progress,
            build_progress_view,
            create_game_progress,
        )

        games = {
            1: advance_game_progress(
                create_game_progress(1, "키움 @ LG", "18:30"),
                step=2,
                status="라인업 대기 · 다음 조회 17:31:00",
            ),
            2: advance_game_progress(
                create_game_progress(2, "두산 @ SSG", "18:30"),
                step=6,
                status="제출 완료",
                model="primary",
                delivery="성공",
            ),
        }
        view = build_progress_view(
            {
                "인증정보 확인": "완료",
                "모델과 메타데이터 로드": "완료",
                "운영 데이터 로드": "진행 중",
                "배포 정보 확인": "대기",
            },
            games,
        )
        console = Console(record=True, width=140)
        console.print(view)
        output = console.export_text()

        self.assertIn("오늘 경기 2 | 완료 1 | 대기 1 | 실패 0", output)
        self.assertIn("키움 @ LG", output)
        self.assertIn("2/6", output)
        self.assertIn("라인업 대기", output)
        self.assertIn("두산 @ SSG", output)
        self.assertIn("primary", output)
        self.assertIn("성공", output)

    def test_display_failure_disables_ui_without_raising(self):
        # Rich 갱신 실패는 화면만 비활성화하고 운영 호출자에게 전파하지 않는다.
        from prediction_progress import PredictionProgressDisplay

        display = PredictionProgressDisplay()
        display._refresh = lambda: (_ for _ in ()).throw(
            RuntimeError("렌더링 실패")
        )

        display.mark_preparation("인증정보 확인", "완료")

        self.assertTrue(display.disabled)

    def test_preparation_context_marks_failure_and_reraises(self):
        # 준비 단계 실패를 화면에 남기면서 원래 오류는 운영 흐름에 전달한다.
        from prediction_progress import PredictionProgressDisplay

        display = PredictionProgressDisplay()

        with self.assertRaisesRegex(ValueError, "모델 오류"):
            with display.preparation("모델과 메타데이터 로드"):
                raise ValueError("모델 오류")

        self.assertEqual(
            display.preparation_states["모델과 메타데이터 로드"],
            "실패",
        )


if __name__ == "__main__":
    unittest.main()
