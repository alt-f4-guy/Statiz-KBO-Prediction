# 실시간 예측 진행 상태표 구현 계획

> **에이전트 작업자 필수 지침:** 각 작업을 순서대로 구현하며
> `superpowers:test-driven-development`를 사용한다. 단계는 체크박스
> (`- [ ]`)로 추적한다.

**목표:** 실시간 예측의 준비 과정과 경기별 라인업 대기, 피처 생성, 모델 추론,
API 제출 상태를 갱신형 Rich 표로 보여준다.

**구조:** 상태 계산과 Rich 렌더링을 새 `prediction_progress.py`에 분리한다.
`predict_2026.py`는 기존 운영 로직을 유지하면서 상태 전이만 보고한다.
상태 계산은 순수 함수로 테스트하고 Rich 출력은 실제 `Console(record=True)`로
렌더링해 검증한다.

**기술 스택:** Python 3.14, `dataclasses`, Rich, Pandas, `unittest`

## 전역 제약

- 확률 계산, 모델 선택, API 엔드포인트와 제출 페이로드를 변경하지 않는다.
- `POLL_SECONDS=60`과 `LINEUP_DEADLINE_MINUTES=30`을 변경하지 않는다.
- 인증값, API 원문 응답과 로컬 비밀 파일 내용을 화면에 출력하지 않는다.
- 화면 출력 실패는 기존 예측·제출 흐름을 중단시키지 않는다.
- 기존 코드 스타일을 따르고 진행 표시 외의 리팩터링을 하지 않는다.
- Pandas 처리에는 `iterrows()`와 `.apply()`를 사용하지 않는다.

---

### 작업 1: 경기 진행 상태와 집계

**파일:**

- 생성: `src/kbo_pipeline/prediction_progress.py`
- 생성: `tests/test_prediction_progress.py`

**인터페이스:**

- 생성: `GameProgress`
- 생성: `create_game_progress(s_no, matchup, start_time) -> GameProgress`
- 생성: `advance_game_progress(progress, *, step, status, model=None, delivery=None, error_type=None) -> GameProgress`
- 생성: `summarize_progress(games) -> ProgressSummary`

- [ ] **1단계: 실패 테스트 작성**

`tests/test_prediction_progress.py`에 다음 테스트를 작성한다.

```python
import unittest


class PredictionProgressTests(unittest.TestCase):
    def test_new_game_starts_at_schedule_check(self):
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
```

- [ ] **2단계: 실패 확인**

실행:

```bash
PYTHONPATH=src/kbo_pipeline:src \
  python3 -m unittest tests.test_prediction_progress -v
```

예상 결과: `prediction_progress` 모듈이 없어 실패한다.

- [ ] **3단계: 최소 상태 모델 구현**

`src/kbo_pipeline/prediction_progress.py`에 다음 상태 모델을 구현한다.

```python
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
    rows = list(games.values())
    completed = sum(row.delivery in {"성공", "만료"} for row in rows)
    failed = sum(row.delivery == "실패" for row in rows)
    return ProgressSummary(
        total=len(rows),
        completed=completed,
        waiting=len(rows) - completed - failed,
        failed=failed,
    )
```

- [ ] **4단계: 상태 테스트 통과 확인**

실행:

```bash
PYTHONPATH=src/kbo_pipeline:src \
  python3 -m unittest tests.test_prediction_progress -v
```

예상 결과: 세 테스트가 모두 통과한다.

- [ ] **5단계: 작업 1 커밋**

```bash
git add src/kbo_pipeline/prediction_progress.py \
  tests/test_prediction_progress.py
git commit -m "feat: 실시간 예측 진행 상태 모델 추가"
```

---

### 작업 2: Rich 상태표와 안전한 화면 갱신

**파일:**

- 수정: `src/kbo_pipeline/prediction_progress.py`
- 수정: `tests/test_prediction_progress.py`

**인터페이스:**

- 소비: `GameProgress`, `ProgressSummary`, `summarize_progress`
- 생성: `build_progress_view(preparation, games, waiting_message="")`
- 생성: `PredictionProgressDisplay`
- 생성: `PredictionProgressDisplay.mark_preparation(name, state)`
- 생성: `PredictionProgressDisplay.preparation(name)`
- 생성: `PredictionProgressDisplay.set_waiting(message)`
- 생성: `PredictionProgressDisplay.register(progress)`
- 생성: `PredictionProgressDisplay.advance(s_no, **changes)`

- [ ] **1단계: 실제 Rich 렌더링 실패 테스트 작성**

`tests/test_prediction_progress.py`에 다음 테스트를 추가한다.

```python
    def test_rendered_view_contains_preparation_summary_and_game_rows(self):
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
        from prediction_progress import PredictionProgressDisplay

        display = PredictionProgressDisplay()
        display._refresh = lambda: (_ for _ in ()).throw(
            RuntimeError("렌더링 실패")
        )

        display.mark_preparation("인증정보 확인", "완료")

        self.assertTrue(display.disabled)

    def test_preparation_context_marks_failure_and_reraises(self):
        from prediction_progress import PredictionProgressDisplay

        display = PredictionProgressDisplay()

        with self.assertRaisesRegex(ValueError, "모델 오류"):
            with display.preparation("모델과 메타데이터 로드"):
                raise ValueError("모델 오류")

        self.assertEqual(
            display.preparation_states["모델과 메타데이터 로드"],
            "실패",
        )
```

- [ ] **2단계: 렌더링 테스트 실패 확인**

실행:

```bash
PYTHONPATH=src/kbo_pipeline:src \
  python3 -m unittest tests.test_prediction_progress -v
```

예상 결과: `build_progress_view`와 `PredictionProgressDisplay`가 없어 실패한다.

- [ ] **3단계: Rich 뷰 구현**

`prediction_progress.py`에 Rich `Group`, `Panel`, `Table`을 사용해 다음 규칙을
구현한다.

```python
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any

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


def build_progress_view(
    preparation: Mapping[str, str],
    games: Mapping[int, GameProgress],
    waiting_message: str = "",
) -> Group:
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
```

`PredictionProgressDisplay`는 `OrderedDict`로 경기 순서를 보존한다.
`start()`는 `Live`를 시작하고 `stop()`은 종료한다. 공개 상태 변경 메서드는
내부 `_safe_refresh()`를 호출하며 `_refresh()`가 예외를 내면 `disabled=True`로
바꾸고 예외를 외부로 전달하지 않는다.

```python
class PredictionProgressDisplay:
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
                build_progress_view(self.preparation_states, self.games),
                console=self.console,
                refresh_per_second=4,
            )
            self.live.start(refresh=True)
        except Exception:
            self.disabled = True

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
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
    def preparation(self, name: str):
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
```

상태 변경 자체의 잘못된 호출은 숨기지 않는다. Rich 갱신에서 발생한 예외만
`_safe_refresh()`가 처리해 상태표 실패가 운영 로직에 영향을 주지 않게 한다.

- [ ] **4단계: Rich 상태표 테스트 통과 확인**

실행:

```bash
PYTHONPATH=src/kbo_pipeline:src \
  python3 -m unittest tests.test_prediction_progress -v
```

예상 결과: 여섯 테스트가 모두 통과하고 출력에 인증값이나 응답 본문이 없다.

- [ ] **5단계: 작업 2 커밋**

```bash
git add src/kbo_pipeline/prediction_progress.py \
  tests/test_prediction_progress.py
git commit -m "feat: 실시간 예측 Rich 상태표 추가"
```

---

### 작업 3: 준비 단계와 경기별 전이를 운영 루프에 연결

**파일:**

- 수정: `scripts/ops/predict_2026.py:1-455`
- 수정: `tests/test_realtime_prediction.py`
- 수정: `docs/pipeline_summary.md`

**인터페이스:**

- 소비: `PredictionProgressDisplay`
- 소비: `create_game_progress`
- 유지: `run_realtime_prediction_system() -> None`

- [ ] **1단계: 경기 표시값 생성 실패 테스트 작성**

운영 루프가 팀 이름과 서울 경기 시각을 일관되게 등록하도록
`scripts/ops/predict_2026.py`에 만들 `_game_progress`의 테스트를
`tests/test_realtime_prediction.py`에 먼저 추가한다.

```python
    def test_game_progress_uses_matchup_and_seoul_start_time(self):
        from predict_2026 import _game_progress

        game = {
            "s_no": 20260496,
            "homeTeam": 5002,
            "awayTeam": 10001,
            "homeTeamName": "LG",
            "awayTeamName": "키움",
        }
        start = pd.Timestamp("2026-07-28T18:30:00+09:00")

        progress = _game_progress(game, start)

        self.assertEqual(progress.s_no, 20260496)
        self.assertEqual(progress.matchup, "키움 @ LG")
        self.assertEqual(progress.start_time, "18:30")
```

- [ ] **2단계: 대상 테스트 실패 확인**

실행:

```bash
PYTHONPATH=src/kbo_pipeline:src:scripts/ops \
  python3 -m unittest \
  tests.test_realtime_prediction.RealtimePredictionTests.test_game_progress_uses_matchup_and_seoul_start_time \
  -v
```

예상 결과: `_game_progress`를 가져올 수 없어 실패한다.

- [ ] **3단계: 준비 상태 표시 연결**

`predict_2026.py`에 다음 import와 도우미를 추가한다.

```python
from datetime import datetime, timedelta

from prediction_progress import (
    GameProgress,
    PredictionProgressDisplay,
    create_game_progress,
)


def _game_progress(
    game: dict[str, Any],
    game_time: pd.Timestamp,
) -> GameProgress:
    home_name = _team_name(game["homeTeam"], game.get("homeTeamName"))
    away_name = _team_name(game["awayTeam"], game.get("awayTeamName"))
    start_time = game_time.tz_convert(SEOUL).strftime("%H:%M")
    return create_game_progress(
        int(game["s_no"]),
        f"{away_name} @ {home_name}",
        start_time,
    )
```

`run_realtime_prediction_system()` 본문을 display 컨텍스트로 감싸고 각 준비
블록은 `display.preparation()`으로 감싼다. 준비 중 예외가 발생하면 해당
항목이 `실패`가 되고 기존 예외가 다시 발생한다. display 컨텍스트는 정상
반환과 예외 모두에서 화면을 종료한다.

```python
with PredictionProgressDisplay() as display:
    with display.preparation("인증정보 확인"):
        credentials = load_api_credentials(require_ptt_idx=True)

    with display.preparation("모델과 메타데이터 로드"):
        model = joblib.load(MODEL_DIR / "best_model.joblib")
        metadata = json.loads(
            (MODEL_DIR / "best_model_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        features = list(model.feature_columns)

    with display.preparation("운영 데이터 로드"):
        games = pd.read_csv(RAW_DATA_DIR / "games_master.csv")
        historical_lineups = pd.read_csv(RAW_DATA_DIR / "lineups.csv")
        rosters = pd.read_csv(RAW_DATA_DIR / "rosters.csv")
        day = pd.read_csv(
            PROCESSED_DATA_DIR / "player_day_processed_v2.csv"
        )
        season = pd.read_csv(
            PROCESSED_DATA_DIR / "player_season_processed_v2.csv"
        )
        training = pd.read_csv(
            FINAL_DATA_DIR / "final_training_set_v9.csv"
        )
        non_draw_training = training.loc[
            training["homeScore"].ne(training["awayScore"])
            & training["year"].lt(2026)
        ]
        league_home_rate = float(
            non_draw_training["homeScore"]
            .gt(non_draw_training["awayScore"])
            .mean()
        )

    with display.preparation("배포 정보 확인"):
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        deployment_context = build_deployment_context(
            PROJECT_ROOT,
            metadata,
            git_commit,
        )

    today = datetime.now(SEOUL)
    day_key = today.strftime("%Y%m%d")
    year = today.strftime("%Y")
    month = today.strftime("%m")
    log_history = _load_prediction_history()
    terminal_s_nos = _terminal_game_ids(log_history)
```

현재 `while True`부터 함수 끝까지의 운영 루프를 위 display 컨텍스트 안에
그대로 들여쓰기하고 다음 단계의 상태 전이 호출만 추가한다.

- [ ] **4단계: 경기별 상태 전이 연결**

기존 루프의 각 경계에 다음 상태 갱신을 추가한다.

```python
# 일정 등록
display.register(_game_progress(game, game_time))

# 경기 시작 이후
display.advance(
    s_no,
    step=6,
    status="경기 시작",
    delivery="만료",
)

# 라인업 조회 직전
display.advance(s_no, step=2, status="라인업 조회")

# 라인업 미완성
next_poll = (
    datetime.now(SEOUL) + timedelta(seconds=POLL_SECONDS)
).strftime("%H:%M:%S")
display.advance(
    s_no,
    step=2,
    status=f"라인업 대기 · 다음 조회 {next_poll}",
)

# 완성 라인업 피처 생성
display.advance(s_no, step=3, status="피처 생성")

# 주 모델 또는 대체 모델 선택
display.advance(
    s_no,
    step=4,
    status=fallback_reason or "모델 추론 완료",
    model=model_type,
)

# 제출 직전
display.advance(s_no, step=5, status="API 제출")

# 제출 실패
display.advance(
    s_no,
    step=6,
    status="제출 실패",
    delivery="실패",
    error_type=exc.__class__.__name__,
)

# 제출 성공
display.advance(
    s_no,
    step=6,
    status=f"제출 완료 · 홈 승률 {probability:.1%}",
    model=model_type,
    delivery="성공",
)
```

일정 조회 실패와 오늘 일정 부재 분기에서는
`display.set_waiting()`으로 사유와 다음 조회 시각을 표시한다. 정상 일정을
받으면 `display.set_waiting("")`으로 해제한다.

기존 `print()` 세 곳은 상태표가 같은 정보를 보존하므로 제거한다.

- [ ] **5단계: 도우미와 기존 실시간 테스트 통과 확인**

실행:

```bash
PYTHONPATH=src/kbo_pipeline:src:scripts/ops \
  python3 -m unittest \
  tests.test_prediction_progress \
  tests.test_realtime_prediction \
  -v
```

예상 결과: 진행 상태와 기존 실시간 예측 테스트가 모두 통과한다.

- [ ] **6단계: 운영 문서 갱신**

`docs/pipeline_summary.md`의 실시간 예측 절에 다음 동작을 기록한다.

```text
실시간 모듈은 준비 4단계와 경기별 6단계 진행 상태를 Rich 표로 갱신한다.
라인업 미발표 시 다음 조회 시각을 표시하고, 각 경기의 사용 모델과
최종 제출 성공·실패·만료 상태를 표에 남긴다. 화면 표시는 확률 계산,
폴링 주기와 제출 계약을 변경하지 않는다.
```

- [ ] **7단계: 전체 검증**

실행:

```bash
PYTHONPATH=src/kbo_pipeline:src:scripts/model:scripts/ops \
  python3 -m unittest discover -s tests -v
```

예상 결과: 전체 테스트가 실패 없이 통과한다.

실행:

```bash
python3 -m compileall -q run_pipeline.py src scripts tests
```

예상 결과: 종료 코드 `0`.

실행:

```bash
git diff --check
```

예상 결과: 출력 없이 종료 코드 `0`.

- [ ] **8단계: 작업 3 커밋**

```bash
git add scripts/ops/predict_2026.py \
  tests/test_realtime_prediction.py \
  docs/pipeline_summary.md
git commit -m "feat: 실시간 예측 단계별 진행 표시"
```
