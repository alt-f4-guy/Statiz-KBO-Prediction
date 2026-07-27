# 일일 운영 진입 흐름 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `python3 run_pipeline.py` 한 줄로 환경설정 검증, 당일 데이터 갱신, 실시간 예측 제출을 실행한다.

**Architecture:** `run_pipeline.py`에 표준 라이브러리 기반 환경파일 로더와 고정된 일일 단계 목록을 둔다. 일일 단계에서는 모델·학습 데이터·배포 메타데이터를 변경하는 모든 학습 및 평가 모듈을 제외한다.

**Tech Stack:** Python 표준 라이브러리, `unittest`, 기존 Rich 터미널 출력

## Global Constraints

- `config/.env`의 기존 셸 환경변수보다 우선순위가 낮아야 한다.
- 일일 실행은 수집 4단계, 원천 정형화, 실시간 예측만 수행한다.
- 튜닝·재학습·백테스트·배포 메타데이터 갱신은 일일 실행에서 제외한다.
- 어느 단계든 실패하면 다음 단계로 진행하지 않는다.

---

### Task 1: 일일 실행 진입점 고정

**Files:**
- Create: `tests/test_run_pipeline.py`
- Modify: `run_pipeline.py`
- Modify: `docs/pipeline_summary.md`

**Interfaces:**
- Produces: `DAILY_PHASES: tuple[tuple[str, str], ...]`
- Produces: `load_runtime_environment(env_path: Path) -> None`
- Consumes: `config/.env`의 `STATIZ_API_KEY`, `STATIZ_SECRET`, `STATIZ_PTT_IDX`

- [ ] **Step 1: 실패하는 단계 목록 테스트 작성**

```python
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
```

- [ ] **Step 2: 실패하는 환경파일 테스트 작성**

```python
def test_environment_file_loads_missing_values_without_overriding_shell(self):
    env_path.write_text(
        "STATIZ_API_KEY=file-key\n"
        "STATIZ_SECRET=file-secret\n"
        "STATIZ_PTT_IDX=file-account\n",
        encoding="utf-8",
    )
    with patch.dict(os.environ, {"STATIZ_API_KEY": "shell-key"}, clear=True):
        run_pipeline.load_runtime_environment(env_path)
        self.assertEqual(os.environ["STATIZ_API_KEY"], "shell-key")
        self.assertEqual(os.environ["STATIZ_SECRET"], "file-secret")
        self.assertEqual(os.environ["STATIZ_PTT_IDX"], "file-account")
```

- [ ] **Step 3: 대상 테스트를 실행해 실패 확인**

Run: `PYTHONPATH=src/kbo_pipeline:src python3 -m unittest tests.test_run_pipeline -v`

Expected: `DAILY_PHASES` 또는 `load_runtime_environment`가 없어서 실패

- [ ] **Step 4: 최소 구현 작성**

`run_pipeline.py`에서 환경파일의 주석·빈 줄을 건너뛰고 `KEY=VALUE`를 읽어
`os.environ.setdefault`으로 설정한다. 세 필수 키가 없으면 `RuntimeError`를
발생시킨다. `main()` 시작 시 로더를 호출하고 기존 `phases` 대신
`DAILY_PHASES`를 순회한다.

- [ ] **Step 5: 운영 문서 갱신**

`docs/pipeline_summary.md`에 `.env` 자동 로드와 다음 일일 실행 명령을 기록한다.

```bash
python3 run_pipeline.py
```

- [ ] **Step 6: 대상 테스트 통과 확인**

Run: `PYTHONPATH=src/kbo_pipeline:src python3 -m unittest tests.test_run_pipeline -v`

Expected: 모든 `test_run_pipeline` 테스트 통과

- [ ] **Step 7: 전체 회귀 검증**

Run: `PYTHONPATH=src/kbo_pipeline:src:scripts/model:scripts/ops python3 -m unittest discover -s tests -v`

Expected: 전체 테스트 통과

Run: `python3 -m compileall -q run_pipeline.py src scripts tests`

Expected: 종료 코드 0

- [ ] **Step 8: 구현 커밋**

```bash
git add run_pipeline.py tests/test_run_pipeline.py docs/pipeline_summary.md
git commit -m "fix: 일일 운영 진입 흐름 고정"
```
