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


if __name__ == "__main__":
    unittest.main()
