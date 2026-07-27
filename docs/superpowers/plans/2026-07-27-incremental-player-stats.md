# 선수 기록 일일 증분 수집 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 당일 예측에 필요한 선수와 연도만 일정 간격으로 수집하고 같은 날 성공분을 재사용한다.

**Architecture:** `player_stats_collection.py`가 당일 경기 팀의 최신 로스터를 벡터화해 선택하고, 스냅샷의 성공일을 기준으로 재사용 가능한 선수-연도를 계산한다. 선수 API 호출은 0.3초 간격을 두며 최종 `StatizAPIError`를 즉시 호출자에게 전달한다.

**Tech Stack:** Python, Pandas, unittest

## Global Constraints

- 서울 날짜 기준 당일 비취소 경기 팀의 최신 로스터만 사용한다.
- 현재 시즌과 직전 시즌만 수집한다.
- 오늘 성공한 현재 시즌과 성공한 종료 시즌은 재사용한다.
- `iterrows()`와 `.apply()`를 사용하지 않는다.
- 선수 API 호출 뒤 0.3초 간격을 둔다.
- 최종 요청 실패는 즉시 파이프라인을 중단한다.

---

### Task 1: 당일 경기 로스터 모집단

**Files:**
- Modify: `src/kbo_pipeline/player_stats_collection.py`
- Modify: `tests/test_player_stats_collection.py`

**Interfaces:**
- Produces: `load_game_day_player_population(roster_path: Path, games_path: Path, target_date: date) -> list[int]`

- [ ] **Step 1: 실패 테스트 작성**

두 팀의 당일 경기, 과거·당일·미래 로스터를 만들고 팀별 당일 이하 최신
로스터의 선수 합집합만 반환하는지 검사한다. 당일 경기가 없으면 빈 목록,
한 팀 로스터가 없으면 `ValueError`인지 별도 검사한다.

- [ ] **Step 2: 적색 테스트 실행**

Run: `PYTHONPATH=src/kbo_pipeline:src python3 -m unittest tests.test_player_stats_collection -v`

Expected: 새 모집단 함수가 없어 실패

- [ ] **Step 3: 벡터화 구현**

경기 날짜와 로스터 날짜를 `pd.to_datetime`으로 변환하고, 당일 팀 코드에
한정한 뒤 `groupby("t_code")["roster_date"].transform("max")`로 최신 행을
선택한다. 누락 팀이 있으면 예외를 발생시킨다.

- [ ] **Step 4: 녹색 테스트 실행**

Run: `PYTHONPATH=src/kbo_pipeline:src python3 -m unittest tests.test_player_stats_collection -v`

Expected: 모집단 테스트 통과

---

### Task 2: 동일일 성공 스냅샷 재사용

**Files:**
- Modify: `src/kbo_pipeline/player_stats_collection.py`
- Modify: `tests/test_player_stats_collection.py`

**Interfaces:**
- Produces: `reusable_player_years(snapshots: pd.DataFrame, current_year: int, target_date: date) -> set[tuple[int, int]]`
- Modifies: `years_to_collect`이 재사용 집합에 있는 현재 시즌도 건너뜀

- [ ] **Step 1: 실패 테스트 작성**

직전 시즌 성공, 오늘 성공한 현재 시즌, 어제 성공한 현재 시즌을 포함한
스냅샷으로 재사용 집합과 요청 연도를 리터럴 값으로 검사한다.

- [ ] **Step 2: 적색 테스트 실행**

Run: `PYTHONPATH=src/kbo_pipeline:src python3 -m unittest tests.test_player_stats_collection -v`

Expected: 오늘 성공분을 구분하지 못해 실패

- [ ] **Step 3: 최소 구현**

성공 행만 선택하고 `fetched_at`을 UTC로 파싱한 뒤 서울 날짜로 변환한다.
종료 시즌은 모든 성공 행, 현재 시즌은 `target_date`와 같은 성공 행만
재사용 집합에 포함한다.

- [ ] **Step 4: 녹색 테스트 실행**

Run: `PYTHONPATH=src/kbo_pipeline:src python3 -m unittest tests.test_player_stats_collection -v`

Expected: 스냅샷 재사용 테스트 통과

---

### Task 3: 호출 간격과 실패 즉시 전파

**Files:**
- Modify: `src/kbo_pipeline/player_stats_collection.py`
- Modify: `scripts/collect/collect_player_stats.py`
- Modify: `tests/test_player_stats_collection.py`
- Modify: `docs/pipeline_summary.md`

**Interfaces:**
- Modifies: `collect_player_snapshots(..., request_interval: float = 0.3, sleep: Callable[[float], None] = time.sleep)`

- [ ] **Step 1: 실패 테스트 작성**

한 선수의 시즌 API 한 번과 일별 API 두 번이 성공할 때 대기값이
`[0.3, 0.3, 0.3]`인지 검사한다. 첫 선수에서 `StatizAPIError`가 발생하면
두 번째 선수를 호출하지 않고 예외가 전달되는지 검사한다.

- [ ] **Step 2: 적색 테스트 실행**

Run: `PYTHONPATH=src/kbo_pipeline:src python3 -m unittest tests.test_player_stats_collection -v`

Expected: 대기 호출이 없고 오류를 삼켜 실패

- [ ] **Step 3: 최소 구현**

각 `api.get` 성공 뒤 주입된 `sleep(request_interval)`을 호출한다.
`StatizAPIError`는 잡지 않고 전달한다. 스크립트는 당일 모집단과
`[current_year - 1, current_year]`를 전달한다.

- [ ] **Step 4: 운영 문서 갱신**

`docs/pipeline_summary.md`에 당일 최신 로스터, 현재·직전 시즌, 동일일 성공
재사용, 0.3초 간격과 실패 중단을 기록한다.

- [ ] **Step 5: 전체 검증**

Run: `PYTHONPATH=src/kbo_pipeline:src:scripts/model:scripts/ops python3 -m unittest discover -s tests -v`

Expected: 전체 테스트 통과

Run: `python3 -m compileall -q run_pipeline.py src scripts tests`

Expected: 종료 코드 0
