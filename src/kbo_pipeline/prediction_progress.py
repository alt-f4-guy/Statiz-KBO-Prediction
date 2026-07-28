"""실시간 예측의 표시 전용 상태와 집계."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping


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
    completed = sum(row.delivery in {"성공", "만료"} for row in rows)
    failed = sum(row.delivery == "실패" for row in rows)
    return ProgressSummary(
        total=len(rows),
        completed=completed,
        waiting=len(rows) - completed - failed,
        failed=failed,
    )
