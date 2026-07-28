# 일정 제한 시 로컬 오프라인 예측 대체 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 일정 API가 요청 제한으로 실패해도 로컬 데이터의 이미 시작한 오늘 경기를 즉시 오프라인 예측하고 진행 화면과 로그에 남긴다.

**Architecture:** `predict_2026.py`에 로컬 시작 경기 선택 함수와 API/로컬 일정 선택 함수를 추가한다. 예측 모듈의 API 인스턴스만 읽기 내부 재시도를 끄고, 기존 경기 처리·모델·제출 시간 게이트는 그대로 재사용한다.

**Tech Stack:** Python 3.14, pandas, unittest, Rich, 기존 Statiz API 클라이언트

## Global Constraints

- 로컬 일정 대체는 서울 기준 오늘이면서 이미 시작한 경기만 반환한다.
- 경기 전 로컬 일정만으로는 `prediction/savePrediction`을 호출하지 않는다.
- 예측 모듈의 `StatizAPI` 인스턴스만 `max_retries=0`을 사용한다.
- 전역 API 기본 재시도와 다른 수집·검증 모듈은 변경하지 않는다.
- 라인업 조회 실패는 기존 `fallback_recent10` 경로로 진행한다.
- 제출 직전 시간 게이트와 제출 최대 세 번 재시도는 변경하지 않는다.
- pandas 필터는 벡터 연산을 사용하고 `iterrows()`와 `.apply()`를 사용하지 않는다.
- `run_pipeline.py`, `tests/test_run_pipeline.py`, `predict_2026_2.py`의 기존 미커밋 변경은 수정하거나 커밋하지 않는다.

---

### Task 1: 로컬에서 이미 시작한 오늘 경기 선택

**Files:**
- Modify: `scripts/ops/predict_2026.py`
- Modify: `tests/test_realtime_prediction.py`

**Interfaces:**
- Consumes: 원천 경기 `pd.DataFrame`, 시간대 포함 `pd.Timestamp`
- Produces: `_local_started_games(games, now_utc) -> list[dict[str, Any]]`

- [ ] **Step 1: 오늘 시작 경기만 반환하는 실패 테스트 작성**

```python
def test_local_started_games_selects_only_started_games_today(self):
    # 로컬 대체는 오늘 시작한 경기만 반환하고 미래·과거 경기는 제외한다.
    from predict_2026 import _local_started_games

    games = pd.DataFrame(
        {
            "s_no": [1, 2, 3],
            "gameDate": [
                1785231000,  # 2026-07-28 18:30 KST
                1785317400,  # 2026-07-29 18:30 KST
                1785144600,  # 2026-07-27 18:30 KST
            ],
            "homeTeam": [5002, 5002, 5002],
            "awayTeam": [10001, 10001, 10001],
        }
    )

    result = _local_started_games(
        games,
        pd.Timestamp("2026-07-28T19:00:00+09:00"),
    )

    self.assertEqual([game["s_no"] for game in result], [1])
```

- [ ] **Step 2: 시작 전과 잘못된 시각을 제외하는 실패 테스트 작성**

```python
def test_local_started_games_excludes_future_and_invalid_start_times(self):
    # 오늘 행이라도 시작 전이거나 시각 파싱이 불가능하면 사용하지 않는다.
    from predict_2026 import _local_started_games

    games = pd.DataFrame(
        {
            "s_no": [1, 2],
            "gameDate": [1785231000, "invalid"],
            "homeTeam": [5002, 5002],
            "awayTeam": [10001, 10001],
        }
    )

    result = _local_started_games(
        games,
        pd.Timestamp("2026-07-28T18:00:00+09:00"),
    )

    self.assertEqual(result, [])
```

- [ ] **Step 3: 테스트가 함수 부재로 실패하는지 확인**

Run:

```bash
PYTHONPATH=src/kbo_pipeline:src:scripts/model:scripts/ops \
  python3 -m unittest \
  tests.test_realtime_prediction.RealtimePredictionTests.test_local_started_games_selects_only_started_games_today \
  tests.test_realtime_prediction.RealtimePredictionTests.test_local_started_games_excludes_future_and_invalid_start_times \
  -v
```

Expected: `_local_started_games` 가져오기 실패로 FAIL

- [ ] **Step 4: 시간대 검증과 벡터 필터 최소 구현**

```python
def _local_started_games(
    games: pd.DataFrame,
    now_utc: pd.Timestamp,
) -> list[dict[str, Any]]:
    """로컬 원천 데이터에서 이미 시작한 오늘 경기만 반환한다."""

    current = pd.Timestamp(now_utc)
    if current.tzinfo is None:
        raise ValueError("로컬 일정 선택에는 시간대 정보가 필요합니다.")

    numeric = pd.to_numeric(games["gameDate"], errors="coerce")
    seconds = numeric.where(numeric.abs().le(1e11), numeric / 1000)
    starts = pd.to_datetime(
        seconds,
        unit="s",
        errors="coerce",
        utc=True,
    )
    current_utc = current.tz_convert("UTC")
    local_dates = starts.dt.tz_convert(SEOUL).dt.date
    mask = (
        starts.notna()
        & local_dates.eq(current.tz_convert(SEOUL).date())
        & starts.le(current_utc)
    )
    return games.loc[mask].to_dict(orient="records")
```

- [ ] **Step 5: Task 1 테스트 통과 확인**

Run:

```bash
PYTHONPATH=src/kbo_pipeline:src:scripts/model:scripts/ops \
  python3 -m unittest tests.test_realtime_prediction -q
```

Expected: 모든 `test_realtime_prediction` 테스트 PASS

- [ ] **Step 6: Task 1 커밋**

```bash
git add scripts/ops/predict_2026.py tests/test_realtime_prediction.py
git commit -m "feat: 로컬 시작 경기 선택 추가"
```

---

### Task 2: 일정 API 실패를 로컬 경기로 즉시 전환

**Files:**
- Modify: `scripts/ops/predict_2026.py`
- Modify: `tests/test_realtime_prediction.py`

**Interfaces:**
- Consumes: `_local_started_games(...)`, `StatizAPI.get(...)`
- Produces: `_load_today_games(api, games, *, now_utc) -> tuple[list[dict[str, Any]], bool]`
- Contract: 반환 튜플의 두 번째 값은 로컬 일정 사용 여부이다.

- [ ] **Step 1: 일정 실패 시 로컬 경기 반환 실패 테스트 작성**

```python
def test_schedule_failure_uses_started_local_games_without_waiting(self):
    # 일정 요청 실패는 외부 sleep 없이 이미 시작한 로컬 경기로 전환한다.
    from predict_2026 import _load_today_games
    from statiz_api import StatizAPIError

    class FailedAPI:
        def get(self, endpoint, params):
            raise StatizAPIError("HTTP 429")

    games = pd.DataFrame(
        {
            "s_no": [1],
            "gameDate": [1785231000],
            "homeTeam": [5002],
            "awayTeam": [10001],
        }
    )

    result, used_local = _load_today_games(
        FailedAPI(),
        games,
        now_utc=pd.Timestamp("2026-07-28T19:00:00+09:00"),
    )

    self.assertTrue(used_local)
    self.assertEqual([game["s_no"] for game in result], [1])
```

- [ ] **Step 2: 로컬 대상도 없으면 원래 오류를 유지하는 실패 테스트 작성**

```python
def test_schedule_failure_without_started_local_games_reraises(self):
    # 경기 전에는 로컬 일정으로 제출하지 않고 일정 오류를 상위 폴링에 전달한다.
    from predict_2026 import _load_today_games
    from statiz_api import StatizAPIError

    class FailedAPI:
        def get(self, endpoint, params):
            raise StatizAPIError("HTTP 429")

    future = pd.DataFrame(
        {
            "s_no": [1],
            "gameDate": [1785231000],
            "homeTeam": [5002],
            "awayTeam": [10001],
        }
    )

    with self.assertRaisesRegex(StatizAPIError, "429"):
        _load_today_games(
            FailedAPI(),
            future,
            now_utc=pd.Timestamp("2026-07-28T18:00:00+09:00"),
        )
```

- [ ] **Step 3: 정상 일정 API 응답 유지 실패 테스트 작성**

```python
def test_successful_schedule_keeps_api_games(self):
    # 정상 응답이 있으면 로컬 데이터가 아니라 API의 오늘 일정을 사용한다.
    from predict_2026 import _load_today_games

    class SuccessfulAPI:
        def get(self, endpoint, params):
            return {
                "20260728": [
                    {
                        "s_no": 10,
                        "homeTeam": 5002,
                        "awayTeam": 10001,
                    }
                ]
            }

    result, used_local = _load_today_games(
        SuccessfulAPI(),
        pd.DataFrame(),
        now_utc=pd.Timestamp("2026-07-28T17:00:00+09:00"),
    )

    self.assertFalse(used_local)
    self.assertEqual([game["s_no"] for game in result], [10])
```

- [ ] **Step 4: 테스트가 함수 부재로 실패하는지 확인**

Run:

```bash
PYTHONPATH=src/kbo_pipeline:src:scripts/model:scripts/ops \
  python3 -m unittest \
  tests.test_realtime_prediction.RealtimePredictionTests.test_schedule_failure_uses_started_local_games_without_waiting \
  tests.test_realtime_prediction.RealtimePredictionTests.test_schedule_failure_without_started_local_games_reraises \
  tests.test_realtime_prediction.RealtimePredictionTests.test_successful_schedule_keeps_api_games \
  -v
```

Expected: `_load_today_games` 가져오기 실패로 FAIL

- [ ] **Step 5: API 일정과 로컬 대체 선택 최소 구현**

```python
def _load_today_games(
    api: StatizAPI,
    games: pd.DataFrame,
    *,
    now_utc: pd.Timestamp,
) -> tuple[list[dict[str, Any]], bool]:
    """오늘 API 일정을 읽고 실패하면 시작한 로컬 경기로 대체한다."""

    current = pd.Timestamp(now_utc)
    local_now = current.tz_convert(SEOUL)
    try:
        schedule = api.get(
            "prediction/gameSchedule",
            {
                "year": local_now.strftime("%Y"),
                "month": local_now.strftime("%m"),
            },
        )
    except StatizAPIError:
        local_games = _local_started_games(games, current)
        if local_games:
            return local_games, True
        raise

    if not isinstance(schedule, dict):
        return [], False
    return list(schedule.get(local_now.strftime("%Y%m%d"), [])), False
```

- [ ] **Step 6: Task 2 테스트 통과 확인**

Run:

```bash
PYTHONPATH=src/kbo_pipeline:src:scripts/model:scripts/ops \
  python3 -m unittest tests.test_realtime_prediction -q
```

Expected: 모든 `test_realtime_prediction` 테스트 PASS

- [ ] **Step 7: Task 2 커밋**

```bash
git add scripts/ops/predict_2026.py tests/test_realtime_prediction.py
git commit -m "feat: 일정 실패 시 로컬 경기로 전환"
```

---

### Task 3: 무재시도 읽기 클라이언트와 진행 메시지 연결

**Files:**
- Modify: `scripts/ops/predict_2026.py`
- Modify: `tests/test_realtime_prediction.py`

**Interfaces:**
- Consumes: `_load_today_games(...)`, 기존 `PredictionProgressDisplay`
- Produces: 일정 실패 즉시 로컬 경기 처리와 완료 메시지

- [ ] **Step 1: 예측 클라이언트가 내부 읽기 재시도를 끄는 실패 테스트 작성**

```python
def test_realtime_system_disables_internal_read_retries(self):
    # 429가 300초 내부 sleep으로 들어가지 않도록 예측 인스턴스만 재시도를 끈다.
    from types import SimpleNamespace
    from unittest.mock import patch

    from prediction_progress import PredictionProgressDisplay
    from predict_2026 import _run_realtime_prediction_system

    captured = {}

    def stop_after_constructor(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("constructor observed")

    with (
        patch(
            "predict_2026.load_api_credentials",
            return_value=SimpleNamespace(
                api_key="key",
                secret="secret",
                ptt_idx="05",
            ),
        ),
        patch("predict_2026.StatizAPI", side_effect=stop_after_constructor),
    ):
        with self.assertRaisesRegex(RuntimeError, "constructor observed"):
            _run_realtime_prediction_system(PredictionProgressDisplay())

    self.assertEqual(captured["max_retries"], 0)
```

- [ ] **Step 2: 테스트가 `max_retries` 누락으로 실패하는지 확인**

Run:

```bash
PYTHONPATH=src/kbo_pipeline:src:scripts/model:scripts/ops \
  python3 -m unittest \
  tests.test_realtime_prediction.RealtimePredictionTests.test_realtime_system_disables_internal_read_retries \
  -v
```

Expected: `captured["max_retries"]` 키 누락으로 FAIL

- [ ] **Step 3: 예측 모듈 인스턴스만 내부 재시도 비활성화**

```python
api = StatizAPI(
    credentials.api_key,
    credentials.secret,
    max_retries=0,
)
```

- [ ] **Step 4: 실시간 루프의 일정 조회를 새 선택 함수로 교체**

```python
while True:
    try:
        today_games, used_local_schedule = _load_today_games(
            api,
            games,
            now_utc=pd.Timestamp.now(tz="UTC"),
        )
    except StatizAPIError as exc:
        next_poll = (
            datetime.now(SEOUL) + timedelta(seconds=POLL_SECONDS)
        ).strftime("%H:%M:%S")
        display.set_waiting(
            f"일정 조회 실패 · {exc.__class__.__name__} · "
            f"다음 조회 {next_poll}"
        )
        time.sleep(POLL_SECONDS)
        continue

    if not today_games:
        next_poll = (
            datetime.now(SEOUL) + timedelta(seconds=POLL_SECONDS)
        ).strftime("%H:%M:%S")
        display.set_waiting(
            f"오늘 일정 대기 · 다음 조회 {next_poll}"
        )
        time.sleep(POLL_SECONDS)
        continue

    display.set_waiting(
        "일정 API 제한 · 로컬 일정으로 오프라인 예측"
        if used_local_schedule
        else ""
    )
```

기존 `day_key`, `year`, `month` 계산과 직접 `api.get(...)` 블록을 제거한다.

- [ ] **Step 5: 완료 메시지를 제출 중립 문구로 변경**

```python
if not pending:
    display.set_waiting("오늘 경기 예측 처리 완료")
    return
```

- [ ] **Step 6: 관련 테스트와 구문 검사**

Run:

```bash
PYTHONPATH=src/kbo_pipeline:src:scripts/model:scripts/ops \
  python3 -m unittest tests.test_realtime_prediction tests.test_prediction_progress -q
python3 -m py_compile scripts/ops/predict_2026.py
```

Expected: 모든 관련 테스트 PASS, 구문 검사 종료 코드 0

- [ ] **Step 7: Task 3 커밋**

```bash
git add scripts/ops/predict_2026.py tests/test_realtime_prediction.py
git commit -m "fix: 일정 제한 대기를 로컬 예측으로 대체"
```

---

### Task 4: 전체 회귀와 실제 터미널 실행 검증

**Files:**
- Verify: `scripts/ops/predict_2026.py`
- Verify: `artifacts/operations/prediction_log.csv`
- Verify: 전체 `tests/`

**Interfaces:**
- Consumes: Tasks 1~3의 최종 구현과 실제 로컬 데이터·모델·인증 환경
- Produces: 자동 테스트 및 실제 실행 증거

- [ ] **Step 1: 전체 구문 검사와 테스트 실행**

Run:

```bash
python3 -m compileall -q run_pipeline.py src scripts tests
PYTHONPATH=src/kbo_pipeline:src:scripts/model:scripts/ops \
  python3 -m unittest discover -s tests -q
```

Expected: 구문 검사 종료 코드 0, 전체 테스트 PASS

- [ ] **Step 2: 기존 프로세스와 로그 상태 확인**

Run:

```bash
ps -axo pid,ppid,etime,stat,command
ls -la artifacts/operations
```

예측 구버전 프로세스가 남아 있으면 정확한 PID와 명령을 확인한 뒤 해당
`run_pipeline.py`와 `scripts.ops.predict_2026` 프로세스에만 `SIGINT`를
보낸다. 다른 Python 프로세스는 종료하지 않는다.

- [ ] **Step 3: 실제 터미널에서 상위 파이프라인 실행**

Run:

```bash
python3 run_pipeline.py
```

PTY가 연결된 세션에서 실행하고 최대 60초 동안 다음 출력을 직접 확인한다.

```text
미제출 예측 완료 · 홈 승률
오늘 경기 예측 처리 완료
```

실제 일정 요청이 429이면
`일정 API 제한 · 로컬 일정으로 오프라인 예측`도 확인한다. 실행 시점에 일정
API가 정상 응답하면 이 문구가 없는 것이 정상이며, 로컬 실패 전환은 Task 2의
결정적 실패 테스트로 별도 검증한다.

실시간 예측 단계가 완료되어 다음 대량 선수 수집 단계가 시작되면 해당
`run_pipeline.py` 프로세스에 `SIGINT`를 보내 후속 API 호출을 중단한다.

- [ ] **Step 4: 실제 로그와 제출 부재 확인**

Run:

```bash
head -n 1 artifacts/operations/prediction_log.csv
tail -n 10 artifacts/operations/prediction_log.csv
```

Expected:

- 오늘 다섯 경기의 `record_type`이 `offline_prediction`
- `api_status`가 `not_submitted`
- `evaluation_role`이 `retrospective_diagnostic`
- `prediction_mode`가 `offline_after_start`
- `delivery` 레코드가 없음

- [ ] **Step 5: 변경 범위 확인**

Run:

```bash
git diff --check
git status --short
git diff master...HEAD --stat
```

Expected: 기능 브랜치 변경은 `scripts/ops/predict_2026.py`,
`tests/test_realtime_prediction.py`와 이 계획 문서뿐이다. 기존 미커밋
`run_pipeline.py`, `tests/test_run_pipeline.py`, `predict_2026_2.py`는 격리
작업공간에 포함되지 않는다.
