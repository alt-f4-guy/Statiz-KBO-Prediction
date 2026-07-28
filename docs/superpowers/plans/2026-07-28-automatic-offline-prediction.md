# 경기 시작 후 자동 오프라인 예측 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 경기 시작 전에는 기존 API 제출을 유지하고, 경기 시작 후에는 자동으로 확률만 계산해 회고 진단 로그와 진행 상태표에 남긴다.

**Architecture:** 기존 실시간 루프의 모델·대체 확률 계산 경로는 공유하되, 경기 완료 경계에서 현재 시각을 다시 검사해 제출 또는 오프라인 기록 중 하나만 수행한다. 오프라인 로그 생성은 작은 순수 함수로 분리하고, 성공 전달과 오프라인 기록만 재시작 완료 상태로 복원한다.

**Tech Stack:** Python 3, pandas, unittest/pytest, Rich, 기존 Statiz API 클라이언트

## Global Constraints

- 경기 시작 후 `prediction/savePrediction`은 호출하지 않는다.
- 경기 시작 후 로그는 `record_type=offline_prediction`, `api_status=not_submitted`, `evaluation_role=retrospective_diagnostic`, `prediction_mode=offline_after_start`를 사용한다.
- 기존 `feature_cutoff_datetime`, 모델 확률 계산, 피처 정의와 대체 모델 선택은 변경하지 않는다.
- 기존 `expired` 로그는 보존하지만 완료 경기 판정에서는 제외한다.
- 성공한 `delivery`와 `offline_prediction`만 재시작 완료 상태로 복원한다.
- 기존 경기 전 제출 페이로드와 재시도 정책은 변경하지 않는다.
- `run_pipeline.py`, `tests/test_run_pipeline.py`, `predict_2026_2.py`는 수정하거나 구현 커밋에 포함하지 않는다.

---

### Task 1: 오프라인 로그 계약과 완료 상태

**Files:**
- Modify: `src/kbo_pipeline/realtime_prediction.py`
- Modify: `scripts/ops/predict_2026.py:126-151`
- Modify: `tests/test_realtime_prediction.py`

**Interfaces:**
- Consumes: 기존 예측 레코드 `dict[str, Any]`
- Produces: `build_offline_prediction_record(prediction, *, recorded_at) -> dict[str, Any]`
- Produces: `_terminal_game_ids(history: pd.DataFrame) -> set[int]`

- [ ] **Step 1: 오프라인 로그 변환 실패 테스트 작성**

```python
def test_offline_record_preserves_probability_and_marks_non_submission(self):
    # 경기 전 예측을 재사용해도 확률·모델은 유지하고 회고 진단으로 격리한다.
    from realtime_prediction import build_offline_prediction_record

    prediction = {
        "recorded_at": "2026-07-28T17:00:00+09:00",
        "record_type": "prediction",
        "s_no": 1,
        "home_win_probability": 0.61,
        "model_type": "primary",
        "evaluation_role": "prospective_holdout",
        "api_status": "pending",
        "error_type": "old",
    }

    result = build_offline_prediction_record(
        prediction,
        recorded_at="2026-07-28T19:00:00+09:00",
    )

    self.assertEqual(result["recorded_at"], "2026-07-28T19:00:00+09:00")
    self.assertEqual(result["record_type"], "offline_prediction")
    self.assertEqual(result["api_status"], "not_submitted")
    self.assertEqual(result["evaluation_role"], "retrospective_diagnostic")
    self.assertEqual(result["prediction_mode"], "offline_after_start")
    self.assertEqual(result["home_win_probability"], 0.61)
    self.assertEqual(result["model_type"], "primary")
    self.assertEqual(result["error_type"], "")
```

- [ ] **Step 2: 완료 경기 복원 실패 테스트 수정**

```python
def test_terminal_games_include_success_and_offline_only(self):
    # 만료 이력은 재처리하고 성공 제출·오프라인 예측만 완료로 복원한다.
    from predict_2026 import _terminal_game_ids

    history = pd.DataFrame(
        {
            "record_type": [
                "delivery",
                "delivery",
                "expired",
                "offline_prediction",
            ],
            "api_status": [
                "success",
                "failed",
                "expired",
                "not_submitted",
            ],
            "s_no": [1, 2, 3, 4],
        }
    )

    self.assertEqual(_terminal_game_ids(history), {1, 4})
```

- [ ] **Step 3: 테스트를 실행해 필요한 동작이 없어 실패하는지 확인**

Run: `pytest -q tests/test_realtime_prediction.py`

Expected: `build_offline_prediction_record` 가져오기 실패와 `_terminal_game_ids`의 `{1, 3}` 결과 때문에 FAIL

- [ ] **Step 4: 오프라인 로그 변환을 최소 구현**

```python
def build_offline_prediction_record(
    prediction_record: dict[str, Any],
    *,
    recorded_at: str,
) -> dict[str, Any]:
    """예측값을 제출하지 않는 경기 후 회고 진단 기록으로 변환한다."""

    return {
        **prediction_record,
        "recorded_at": recorded_at,
        "record_type": "offline_prediction",
        "api_status": "not_submitted",
        "error_type": "",
        "evaluation_role": "retrospective_diagnostic",
        "prediction_mode": "offline_after_start",
    }
```

- [ ] **Step 5: 완료 경기 판정을 성공 전달과 오프라인 예측으로 제한**

```python
rows = history.loc[
    history["record_type"].eq("offline_prediction")
    | (
        history["record_type"].eq("delivery")
        & history.get(
            "api_status", pd.Series(index=history.index, dtype="object")
        ).eq("success")
    ),
    "s_no",
]
```

- [ ] **Step 6: 단위 테스트 통과 확인**

Run: `pytest -q tests/test_realtime_prediction.py`

Expected: PASS

- [ ] **Step 7: Task 1 커밋**

```bash
git add src/kbo_pipeline/realtime_prediction.py scripts/ops/predict_2026.py tests/test_realtime_prediction.py
git commit -m "feat: 오프라인 예측 로그 계약 추가"
```

---

### Task 2: 경기 시각에 따른 제출·오프라인 완료 자동 전환

**Files:**
- Modify: `scripts/ops/predict_2026.py:210-549`
- Modify: `tests/test_realtime_prediction.py`

**Interfaces:**
- Consumes: `build_offline_prediction_record(...)`, `prediction_window_is_open(...)`, 기존 `send_prediction_with_retry(...)`
- Produces: `_complete_prediction(api, display, prediction_record, payload, *, game_time, now_utc) -> tuple[dict[str, Any], bool]`
- Produces: `_lineup_wait_required(*, submit_before_start, complete, now_utc, deadline) -> bool`
- Contract: 반환 튜플의 두 번째 값은 재시작 시 완료 처리할지 여부이다.

- [ ] **Step 1: 경기 시작 후 제출 0회와 오프라인 기록 실패 테스트 작성**

```python
def test_complete_prediction_after_start_records_offline_without_api_call(self):
    # 경기 시작 후에는 확률을 남기되 저장 API를 한 번도 호출하지 않는다.
    from unittest.mock import patch

    from predict_2026 import _complete_prediction

    class CountingAPI:
        def __init__(self):
            self.post_calls = 0

        def post(self, endpoint, payload):
            self.post_calls += 1
            return {"result_cd": 100}

    class Display:
        def __init__(self):
            self.changes = []

        def advance(self, s_no, **changes):
            self.changes.append((s_no, changes))

    api = CountingAPI()
    display = Display()
    prediction = {
        "recorded_at": "2026-07-28T17:00:00+09:00",
        "record_type": "prediction",
        "s_no": 1,
        "game_datetime": "2026-07-28T18:30:00+09:00",
        "home_win_probability": 0.61,
        "model_type": "primary",
        "evaluation_role": "prospective_holdout",
        "api_status": "pending",
    }

    with patch("predict_2026.append_prediction_log") as append:
        record, terminal = _complete_prediction(
            api,
            display,
            prediction,
            {"s_no": 1},
            game_time=pd.Timestamp("2026-07-28T18:30:00+09:00"),
            now_utc=pd.Timestamp("2026-07-28T19:00:00+09:00"),
        )

    self.assertEqual(api.post_calls, 0)
    self.assertTrue(terminal)
    self.assertEqual(record["record_type"], "offline_prediction")
    self.assertEqual(record["api_status"], "not_submitted")
    append.assert_called_once()
    self.assertEqual(display.changes[-1][1]["delivery"], "미제출")
```

- [ ] **Step 2: 경기 시작 전 성공 제출 회귀 테스트 작성**

```python
def test_complete_prediction_before_start_keeps_successful_submission(self):
    # 경기 전에는 기존 저장 API와 성공 delivery 기록을 유지한다.
    from unittest.mock import patch

    from predict_2026 import _complete_prediction

    class SuccessfulAPI:
        def __init__(self):
            self.post_calls = 0

        def post(self, endpoint, payload):
            self.post_calls += 1
            return {"result_cd": 100}

    class Display:
        def advance(self, s_no, **changes):
            pass

    api = SuccessfulAPI()
    prediction = {
        "recorded_at": "2026-07-28T17:00:00+09:00",
        "record_type": "prediction",
        "s_no": 1,
        "game_datetime": "2026-07-28T18:30:00+09:00",
        "home_win_probability": 0.61,
        "model_type": "primary",
        "api_status": "pending",
    }

    with patch("predict_2026.append_prediction_log") as append:
        record, terminal = _complete_prediction(
            api,
            Display(),
            prediction,
            {"s_no": 1},
            game_time=pd.Timestamp("2026-07-28T18:30:00+09:00"),
            now_utc=pd.Timestamp("2026-07-28T18:00:00+09:00"),
        )

    self.assertEqual(api.post_calls, 1)
    self.assertTrue(terminal)
    self.assertEqual(record["record_type"], "delivery")
    self.assertEqual(record["api_status"], "success")
    append.assert_called_once()
```

- [ ] **Step 3: 경기 후 불완전 라인업도 예측 경로로 진행하는 실패 테스트 작성**

```python
def test_incomplete_lineup_after_start_does_not_wait(self):
    # 경기 후에는 다음 폴링을 기다리지 않고 대체 확률 계산으로 진행한다.
    from predict_2026 import _lineup_wait_required

    start = pd.Timestamp("2026-07-28T18:30:00+09:00")

    self.assertFalse(
        _lineup_wait_required(
            submit_before_start=False,
            complete=False,
            now_utc=start + pd.Timedelta(minutes=30),
            deadline=start - pd.Timedelta(minutes=30),
        )
    )
```

- [ ] **Step 4: 테스트를 실행해 완료 경계와 대기 판정이 없어 실패하는지 확인**

Run: `pytest -q tests/test_realtime_prediction.py`

Expected: `_complete_prediction`과 `_lineup_wait_required` 가져오기 실패로 FAIL

- [ ] **Step 5: 오프라인 변환 함수를 운영 모듈에 연결**

```python
from realtime_prediction import (
    append_prediction_log,
    build_delivery_record,
    build_offline_prediction_record,
    build_prediction_payload,
    # 기존 import는 그대로 유지
)
```

- [ ] **Step 6: 제출과 오프라인 기록을 나누는 완료 경계 최소 구현**

```python
def _complete_prediction(
    api: StatizAPI,
    display: PredictionProgressDisplay,
    prediction_record: dict[str, Any],
    payload: dict[str, Any],
    *,
    game_time: pd.Timestamp,
    now_utc: pd.Timestamp,
) -> tuple[dict[str, Any], bool]:
    s_no = int(prediction_record["s_no"])
    probability = float(prediction_record["home_win_probability"])
    model_type = str(prediction_record["model_type"])

    if not prediction_window_is_open(now_utc, game_time):
        record = build_offline_prediction_record(
            prediction_record,
            recorded_at=datetime.now(SEOUL).isoformat(),
        )
        append_prediction_log(PREDICTION_LOG, record)
        display.advance(
            s_no,
            step=6,
            status=f"미제출 예측 완료 · 홈 승률 {probability:.1%}",
            model=model_type,
            delivery="미제출",
        )
        return record, True

    display.advance(s_no, step=5, status="API 제출")
    try:
        send_prediction_with_retry(api, payload)
    except StatizAPIError as exc:
        record = build_delivery_record(
            prediction_record,
            recorded_at=datetime.now(SEOUL).isoformat(),
            api_status="failed",
            error_type=exc.__class__.__name__,
        )
        append_prediction_log(PREDICTION_LOG, record)
        display.advance(
            s_no,
            step=6,
            status="제출 실패",
            delivery="실패",
            error_type=exc.__class__.__name__,
        )
        return record, False

    record = build_delivery_record(
        prediction_record,
        recorded_at=datetime.now(SEOUL).isoformat(),
        api_status="success",
    )
    append_prediction_log(PREDICTION_LOG, record)
    display.advance(
        s_no,
        step=6,
        status=f"제출 완료 · 홈 승률 {probability:.1%}",
        model=model_type,
        delivery="성공",
    )
    return record, True
```

- [ ] **Step 7: 라인업 대기 조건을 명시적인 순수 함수로 구현**

```python
def _lineup_wait_required(
    *,
    submit_before_start: bool,
    complete: bool,
    now_utc: pd.Timestamp,
    deadline: pd.Timestamp,
) -> bool:
    """경기 전 라인업 마감 전일 때만 다음 폴링을 기다린다."""

    return submit_before_start and not complete and now_utc < deadline
```

- [ ] **Step 8: 실시간 루프의 조기 만료 분기를 자동 모드 선택으로 교체**

```python
now_utc = pd.Timestamp.now(tz="UTC")
submit_before_start = prediction_window_is_open(now_utc, game_time)
existing = _existing_prediction(log_history, s_no)
```

기존 `expired_record` 생성과 `continue`를 제거한다. 라인업 대기 조건은 경기 전
경로에만 적용한다.

```python
if _lineup_wait_required(
    submit_before_start=submit_before_start,
    complete=complete,
    now_utc=pd.Timestamp.now(tz="UTC"),
    deadline=deadline,
):
    # 기존 라인업 대기 표시 후 다음 경기로 이동
    continue
```

- [ ] **Step 9: 새 예측 로그는 경기 전에만 기록하고 완료 경계를 호출**

```python
if submit_before_start:
    append_prediction_log(PREDICTION_LOG, prediction_record)
    log_history = pd.concat(
        [log_history, pd.DataFrame([prediction_record])],
        ignore_index=True,
    )

record, terminal = _complete_prediction(
    api,
    display,
    prediction_record,
    payload,
    game_time=game_time,
    now_utc=pd.Timestamp.now(tz="UTC"),
)
log_history = pd.concat(
    [log_history, pd.DataFrame([record])],
    ignore_index=True,
)
if terminal:
    terminal_s_nos.add(s_no)
```

경기 시작 전 생성된 미제출 `prediction`이 있으면 기존 `_existing_prediction`
결과를 그대로 `_complete_prediction`에 전달해 같은 확률을 재사용한다.

- [ ] **Step 10: 자동 전환과 기존 제출 테스트 통과 확인**

Run: `pytest -q tests/test_realtime_prediction.py`

Expected: PASS

- [ ] **Step 11: Task 2 커밋**

```bash
git add scripts/ops/predict_2026.py tests/test_realtime_prediction.py
git commit -m "feat: 경기 후 오프라인 예측으로 자동 전환"
```

---

### Task 3: 진행 상태와 평가 격리 회귀 검증

**Files:**
- Modify: `src/kbo_pipeline/prediction_progress.py`
- Modify: `tests/test_prediction_progress.py`
- Modify: `tests/test_prediction_log_evaluation.py`

**Interfaces:**
- Consumes: `GameProgress.delivery`
- Produces: `summarize_progress(...)`에서 `미제출`을 완료로 집계
- Contract: `prepare_evaluation_rows(...)`는 `offline_prediction`을 반환하지 않음

- [ ] **Step 1: 미제출 완료 집계 실패 테스트 작성**

```python
def test_summary_counts_offline_prediction_as_completed(self):
    # 의도적으로 제출하지 않은 경기 후 예측은 실패가 아니라 완료다.
    from prediction_progress import (
        advance_game_progress,
        create_game_progress,
        summarize_progress,
    )

    offline = advance_game_progress(
        create_game_progress(1, "원정 @ 홈", "18:30"),
        step=6,
        status="미제출 예측 완료 · 홈 승률 61.0%",
        delivery="미제출",
    )

    summary = summarize_progress({1: offline})

    self.assertEqual(summary.completed, 1)
    self.assertEqual(summary.waiting, 0)
    self.assertEqual(summary.failed, 0)
```

- [ ] **Step 2: 오프라인 로그 평가 제외 회귀 테스트 작성**

```python
def test_offline_prediction_is_excluded_from_prospective_evaluation(self):
    # 필드가 충분해도 회고 진단 레코드는 전향 평가에 들어가지 않는다.
    from evaluate_prediction_log import prepare_evaluation_rows

    log = _prediction_log()
    offline = log.iloc[[0]].copy()
    offline["record_type"] = "offline_prediction"
    offline["s_no"] = 4
    offline["recorded_at"] = "2026-07-29T19:00:00+09:00"
    offline["game_datetime"] = "2026-07-29T18:30:00+09:00"
    offline["evaluation_role"] = "retrospective_diagnostic"
    offline["api_status"] = "not_submitted"
    log = pd.concat([log, offline], ignore_index=True)

    result = prepare_evaluation_rows(
        log,
        _games(),
        deployment_id="deploy-a",
    )

    self.assertNotIn(4, result["s_no"].tolist())
```

- [ ] **Step 3: 진행 집계 테스트가 실패하고 평가 테스트가 통과하는지 확인**

Run: `pytest -q tests/test_prediction_progress.py tests/test_prediction_log_evaluation.py`

Expected: 진행 집계는 완료 0으로 FAIL, 평가 제외는 PASS

- [ ] **Step 4: 미제출을 완료 상태에 추가**

```python
completed = sum(
    row.delivery in {"성공", "만료", "미제출"} for row in rows
)
```

기존 `만료` 표시는 과거 상태 렌더링 호환을 위해 집계에서 유지한다. 실시간
루프는 새 `만료` 로그를 생성하지 않는다.

- [ ] **Step 5: 관련 테스트 전체 통과 확인**

Run: `pytest -q tests/test_prediction_progress.py tests/test_prediction_log_evaluation.py tests/test_realtime_prediction.py`

Expected: PASS

- [ ] **Step 6: Task 3 커밋**

```bash
git add src/kbo_pipeline/prediction_progress.py tests/test_prediction_progress.py tests/test_prediction_log_evaluation.py
git commit -m "test: 오프라인 예측 상태와 평가 격리 고정"
```

---

### Task 4: 전체 회귀 검증과 문서 일치 확인

**Files:**
- Verify: `scripts/ops/predict_2026.py`
- Verify: `src/kbo_pipeline/realtime_prediction.py`
- Verify: `src/kbo_pipeline/prediction_progress.py`
- Verify: 전체 `tests/`

**Interfaces:**
- Consumes: Tasks 1~3의 최종 구현
- Produces: 구문 검사와 전체 테스트가 통과한 병합 가능 커밋 묶음

- [ ] **Step 1: 변경 파일 구문 검사**

Run: `python -m py_compile scripts/ops/predict_2026.py src/kbo_pipeline/realtime_prediction.py src/kbo_pipeline/prediction_progress.py`

Expected: 출력 없이 종료 코드 0

- [ ] **Step 2: 전체 테스트 실행**

Run: `pytest -q`

Expected: 모든 테스트 PASS

- [ ] **Step 3: 요청 범위 밖 변경이 없는지 확인**

Run: `git diff --check && git status --short && git diff master...HEAD --stat`

Expected: 공백 오류 없음. 변경 파일은 계획에 명시한 운영 코드·테스트와 이 계획 문서뿐이며, 기존 미커밋 파일은 격리 작업공간에 나타나지 않음.

- [ ] **Step 4: 실제 분기 안전성 점검**

Run: `rg -n "expired_record|send_prediction_with_retry|offline_prediction|not_submitted|미제출" scripts/ops/predict_2026.py src/kbo_pipeline tests`

Expected: 실시간 루프에 `expired_record` 생성이 없고, 제출 호출은 `_complete_prediction`의 경기 전 분기 안에만 존재하며, 오프라인 로그·표시·테스트가 모두 연결됨.
