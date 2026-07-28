"""실시간 예측의 표시 전용 상태와 집계."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Iterator, Mapping

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


PREPARATION_STEPS = (
    "인증정보 확인",
    "모델과 메타데이터 로드",
    "운영 데이터 로드",
    "배포 정보 확인",
)


@dataclass(frozen=True)
class GameProgress:
    s_no: int
    matchup: str
    start_time: str
    step: int = 1
    status: str = "경기 확인"
    model: str = "-"
    delivery: str = "대기"
    error_type: str = ""


@dataclass(frozen=True)
class ProgressSummary:
    total: int
    completed: int
    waiting: int
    failed: int


def create_game_progress(
    s_no: int,
    matchup: str,
    start_time: str,
) -> GameProgress:
    """새 경기의 초기 표시 상태를 만든다."""

    return GameProgress(
        s_no=int(s_no),
        matchup=matchup,
        start_time=start_time,
    )


def advance_game_progress(
    progress: GameProgress,
    *,
    step: int,
    status: str,
    model: str | None = None,
    delivery: str | None = None,
    error_type: str | None = None,
) -> GameProgress:
    """진행 단계를 뒤로 돌리지 않고 새 표시 상태를 반환한다."""

    if step < progress.step:
        return progress
    return replace(
        progress,
        step=step,
        status=status,
        model=progress.model if model is None else model,
        delivery=progress.delivery if delivery is None else delivery,
        error_type=(
            progress.error_type if error_type is None else error_type
        ),
    )


def summarize_progress(
    games: Mapping[int, GameProgress],
) -> ProgressSummary:
    """경기 상태를 완료·대기·실패로 집계한다."""

    rows = list(games.values())
    completed = sum(
        row.delivery in {"성공", "만료", "미제출"} for row in rows
    )
    failed = sum(row.delivery == "실패" for row in rows)
    return ProgressSummary(
        total=len(rows),
        completed=completed,
        waiting=len(rows) - completed - failed,
        failed=failed,
    )


def build_progress_view(
    preparation: Mapping[str, str],
    games: Mapping[int, GameProgress],
    waiting_message: str = "",
) -> Group:
    """준비 상태, 전체 집계와 경기별 상태표를 만든다."""

    prep_text = " | ".join(
        f"{name}: {preparation.get(name, '대기')}"
        for name in PREPARATION_STEPS
    )
    summary = summarize_progress(games)
    headline = (
        f"오늘 경기 {summary.total} | 완료 {summary.completed} | "
        f"대기 {summary.waiting} | 실패 {summary.failed}"
    )
    if waiting_message:
        headline = f"{headline}\n{waiting_message}"

    table = Table(header_style="bold magenta", expand=True)
    table.add_column("경기")
    table.add_column("시작", width=8)
    table.add_column("진행", width=7, justify="center")
    table.add_column("현재 상태")
    table.add_column("모델", width=18)
    table.add_column("제출", width=8, justify="center")
    for row in games.values():
        status = row.status
        if row.error_type:
            status = f"{status} · {row.error_type}"
        table.add_row(
            row.matchup,
            row.start_time,
            f"{row.step}/6",
            status,
            row.model,
            row.delivery,
        )
    return Group(
        Panel(prep_text, title="실시간 예측 준비"),
        Panel(headline, title="전체 진행"),
        table,
    )


class PredictionProgressDisplay:
    """Rich 상태표를 갱신하되 화면 오류를 운영 흐름과 격리한다."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self.preparation_states = {
            name: "대기" for name in PREPARATION_STEPS
        }
        self.games: OrderedDict[int, GameProgress] = OrderedDict()
        self.waiting_message = ""
        self.live: Live | None = None
        self.disabled = False

    def _safe_refresh(self) -> None:
        if self.disabled:
            return
        try:
            self._refresh()
        except Exception:
            self.disabled = True

    def _refresh(self) -> None:
        if self.live is not None:
            self.live.update(
                build_progress_view(
                    self.preparation_states,
                    self.games,
                    self.waiting_message,
                ),
                refresh=True,
            )

    def start(self) -> None:
        if self.disabled or self.live is not None:
            return
        try:
            self.live = Live(
                build_progress_view(
                    self.preparation_states,
                    self.games,
                ),
                console=self.console,
                refresh_per_second=4,
            )
            self.live.start(refresh=True)
        except Exception:
            self.disabled = True

    def __enter__(self) -> PredictionProgressDisplay:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        self.stop()

    def stop(self) -> None:
        if self.live is None:
            return
        try:
            self.live.stop()
        except Exception:
            self.disabled = True
        finally:
            self.live = None

    def mark_preparation(self, name: str, state: str) -> None:
        self.preparation_states[name] = state
        self._safe_refresh()

    @contextmanager
    def preparation(self, name: str) -> Iterator[None]:
        """준비 단계의 진행·완료·실패 상태를 일관되게 표시한다."""

        self.mark_preparation(name, "진행 중")
        try:
            yield
        except Exception:
            self.mark_preparation(name, "실패")
            raise
        else:
            self.mark_preparation(name, "완료")

    def set_waiting(self, message: str) -> None:
        self.waiting_message = message
        self._safe_refresh()

    def register(self, progress: GameProgress) -> None:
        self.games.setdefault(progress.s_no, progress)
        self._safe_refresh()

    def advance(self, s_no: int, **changes: Any) -> None:
        current = self.games[s_no]
        self.games[s_no] = advance_game_progress(current, **changes)
        self._safe_refresh()
