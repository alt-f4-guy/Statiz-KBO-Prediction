# 일정 증분 수집과 요청 제한 처리 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 일일 일정 API 호출을 현재 월과 미종료 경기 월로 제한하고 HTTP 429의 서버 쿨다운을 준수한다.

**Architecture:** `collect_schedule.py`가 기존 일정 데이터에서 요청할 연월을 벡터화해 계산한다. `StatizAPI`는 429 응답의 구조화된 쿨다운 값을 읽어 같은 요청을 재시도하며, 월 요청이 최종 실패하면 일정 수집 전체가 실패한다.

**Tech Stack:** Python, Pandas, Requests, unittest

## Global Constraints

- 기존 데이터가 없을 때만 2023년 1월부터 전체 월을 요청한다.
- 기존 데이터가 있으면 현재 월과 점수가 비어 있고 취소 상태 `4`가 아닌
  과거 월만 요청한다.
- 미래 월은 요청하지 않는다.
- `iterrows()`와 `.apply()`를 사용하지 않는다.
- 429 쿨다운은 최대 300초로 제한하고 인증값을 로그에 남기지 않는다.
- 한 월이라도 최종 실패하면 파일을 저장하지 않고 수집 단계를 실패시킨다.

---

### Task 1: HTTP 429 서버 쿨다운 준수

**Files:**
- Modify: `src/kbo_pipeline/statiz_api.py`
- Modify: `tests/test_configuration.py`

**Interfaces:**
- Consumes: HTTP 429 JSON의 `rate_limit.cooldown_sec`
- Produces: `StatizAPI._rate_limit_delay(response, fallback) -> float`

- [ ] **Step 1: 실패하는 쿨다운 테스트 작성**

429 응답 뒤 성공 응답을 반환하는 가짜 세션과 대기시간 수집 함수를 사용한다.
`cooldown_sec=53`일 때 `sleep` 호출값이 `[53.0]`인지 검사한다.

- [ ] **Step 2: 적색 테스트 실행**

Run: `PYTHONPATH=src/kbo_pipeline:src python3 -m unittest tests.test_configuration.ConfigurationTests.test_http_429_uses_server_cooldown_before_retry -v`

Expected: 기존 구현이 1초만 기다려 `53.0` 기대값과 달라 실패

- [ ] **Step 3: 최소 쿨다운 구현**

응답 JSON에서 양수 쿨다운을 읽어 `[0, 300]` 범위로 제한한다. JSON이
아니거나 필드가 없으면 기존 지수 백오프 값을 사용한다. 경고 로그에는
대기시간만 기록한다.

- [ ] **Step 4: 녹색 테스트 실행**

Run: `PYTHONPATH=src/kbo_pipeline:src python3 -m unittest tests.test_configuration -v`

Expected: 설정과 HTTP 클라이언트 테스트 전체 통과

---

### Task 2: 요청 연월 증분 선택

**Files:**
- Modify: `scripts/collect/collect_schedule.py`
- Create: `tests/test_schedule_collection.py`

**Interfaces:**
- Produces: `schedule_months_to_fetch(existing: pd.DataFrame, now: datetime) -> list[tuple[str, str]]`

- [ ] **Step 1: 실패하는 월 선택 테스트 작성**

다음 세 사례를 리터럴 기대값으로 검사한다.

- 완결된 기존 데이터: 현재 월 하나
- 미종료 과거 경기: 과거 월과 현재 월
- 빈 데이터: 2023년 1월부터 현재 월까지

- [ ] **Step 2: 적색 테스트 실행**

Run: `PYTHONPATH=src/kbo_pipeline:src python3 -m unittest tests.test_schedule_collection.ScheduleCollectionTests -v`

Expected: `schedule_months_to_fetch`가 없어 실패

- [ ] **Step 3: 벡터화된 월 선택 구현**

`year`, `month`, `state`, `homeScore`, `awayScore`를 벡터화해 숫자로
변환하고, 점수가 하나라도 비어 있으며 취소 상태 `4`가 아닌 행의 연월을
현재 월과 합친다. 빈 데이터에서는
`pd.period_range("2023-01", current_period, freq="M")`를 사용한다.

- [ ] **Step 4: 녹색 테스트 실행**

Run: `PYTHONPATH=src/kbo_pipeline:src python3 -m unittest tests.test_schedule_collection.ScheduleCollectionTests -v`

Expected: 월 선택 테스트 전체 통과

---

### Task 3: 수집 실패 전파와 운영 문서

**Files:**
- Modify: `scripts/collect/collect_schedule.py`
- Modify: `tests/test_schedule_collection.py`
- Modify: `docs/pipeline_summary.md`

**Interfaces:**
- Consumes: `schedule_months_to_fetch`
- Produces: 월 API 실패 시 `StatizAPIError`를 호출자에게 전달

- [ ] **Step 1: 실패 전파 테스트 작성**

기존 CSV와 항상 `StatizAPIError`를 발생시키는 가짜 API를 사용해
`run_schedule_collection()`이 예외를 전달하는지 검사한다.

- [ ] **Step 2: 적색 테스트 실행**

Run: `PYTHONPATH=src/kbo_pipeline:src python3 -m unittest tests.test_schedule_collection.ScheduleCollectionTests.test_month_failure_stops_collection -v`

Expected: 기존 구현이 예외를 삼켜 테스트 실패

- [ ] **Step 3: 증분 루프와 실패 전파 구현**

전체 연월 중첩 루프를 선택된 연월 루프로 교체하고 `StatizAPIError`를 잡아
계속하지 않는다. 모든 선택 월이 성공한 뒤에만 기존 병합과 원자적 저장을
수행한다.

- [ ] **Step 4: 문서 갱신**

`docs/pipeline_summary.md`의 일정 수집 설명에 최초 전체 수집, 이후 현재 월과
미종료 월 수집, 429 서버 쿨다운 준수를 기록한다.

- [ ] **Step 5: 전체 검증**

Run: `PYTHONPATH=src/kbo_pipeline:src:scripts/model:scripts/ops python3 -m unittest discover -s tests -v`

Expected: 전체 테스트 통과

Run: `python3 -m compileall -q run_pipeline.py src scripts tests`

Expected: 종료 코드 0
