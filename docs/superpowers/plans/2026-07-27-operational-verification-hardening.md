# KBO 운영 검증 강화 구현 계획

> **에이전트 작업자용:** 이 계획은 `superpowers:executing-plans`를 사용하여 작업별로 실행한다. 각 단계는 체크박스로 진행 상태를 기록한다.

**목표:** 2026-07-28 운영 재개 전에 경기 후 예측을 차단하고, 전향적 표본·운영 로그·통계 판정·API 읽기 계약·문서를 현재 저장소 구조에 맞게 검증한다.

**구조:** 기존 CSV 추가 기록과 `scripts/`·`src/kbo_pipeline/` 구조를 유지한다. 배포 감사 정보는 작은 순수 함수로 계산하고, 실시간 스크립트는 예측·전송 이벤트를 기록하며, 평가기는 승인된 배포의 전향적 경기만 집계한다.

**기술 구성:** Python 3.14, Pandas, NumPy, scikit-learn, Statiz API, `unittest`, CSV/JSON

## 전역 제약

- 모든 시각은 `Asia/Seoul` 기준으로 기록하고 비교한다.
- API 키, 비밀키, 서명 헤더와 전체 API 응답을 파일에 저장하지 않는다.
- `prediction/savePrediction.percent`는 항상 `P(Home) × 100`이다.
- 대상 경기가 없는 2026-07-27에는 쓰기 API를 호출하지 않는다.
- 모델 재학습, 하이퍼파라미터 변경과 피처 정의 변경은 범위에서 제외한다.
- 통계 재표본 시드는 42로 고정한다.
- 기존 운영 로그는 수정하지 않고 새 행만 추가한다.
- 소스 수정은 실패 테스트를 먼저 확인한 뒤 최소 구현으로 통과시킨다.

---

### 작업 1: 배포 감사 정보와 상대 경로

**파일**

- 생성: `src/kbo_pipeline/deployment.py`
- 수정: `scripts/model/compare_models.py`
- 수정: `artifacts/models/best_model_metadata.json`
- 생성: `tests/test_deployment.py`
- 수정: `tests/test_model_comparison.py`

**인터페이스**

- 생성: `sha256_file(path: Path) -> str`
- 생성: `build_deployment_context(project_root: Path, metadata: Mapping[str, Any], git_commit: str) -> dict[str, Any]`
- 생성: `evaluation_role_for(recorded_at: Timestamp, prospective_start_date: str) -> str`
- `build_deployment_context` 출력: `deployment_id`, `git_commit`, 세 체크섬, `prospective_start_date`, `baseline_home_probability`

- [ ] **1단계: 상대 경로·고정 기준선·재현 가능한 배포 식별자 실패 테스트 작성**

```python
def test_deployment_context_is_reproducible_and_uses_relative_paths(self):
    context_a = build_deployment_context(root, metadata, "abc123")
    context_b = build_deployment_context(root, metadata, "abc123")
    self.assertEqual(context_a, context_b)
    self.assertNotIn(str(root), metadata["split_manifest"])

def test_evaluation_role_starts_on_registered_seoul_date(self):
    self.assertEqual(
        evaluation_role_for(
            pd.Timestamp("2026-07-28T00:00:00+09:00"),
            "2026-07-28",
        ),
        "prospective_holdout",
    )
```

- [ ] **2단계: 실패 확인**

실행:

```bash
PYTHONPATH=src/kbo_pipeline:src:scripts/model \
  python3 -m unittest tests.test_deployment tests.test_model_comparison -v
```

예상: `deployment` 모듈 또는 새 메타데이터 필드가 없어 실패한다.

- [ ] **3단계: 최소 배포 감사 구현**

`deployment.py`는 모델·v9 데이터·시간 분할 명세를 SHA-256으로 읽고, 정렬된 감사 필드 JSON의 SHA-256 앞 16자를 `deployment_id`로 사용한다. `compare_models.py`는 다음 상대 경로와 고정 기준선을 기록한다.

```python
metadata = {
    # 기존 선택 정보 유지
    "data_version": "final_training_set_v9",
    "data_path": "data/final/final_training_set_v9.csv",
    "model_path": "artifacts/models/best_model.joblib",
    "split_manifest": "data/final/time_split_manifest.json",
    "baseline_home_probability": float(baseline_probability),
}
```

로컬 운영 메타데이터에는 `prospective_start_date: "2026-07-28"`을 추가한다.

- [ ] **4단계: 대상 테스트 통과 확인**

실행: 2단계와 동일  
예상: 모든 테스트 통과

- [ ] **5단계: 작업 1 커밋**

```bash
git add src/kbo_pipeline/deployment.py scripts/model/compare_models.py \
  tests/test_deployment.py tests/test_model_comparison.py
git commit -m "feat: 배포 감사 정보 고정"
```

---

### 작업 2: 경기 시작 시각 게이트와 운영 로그

**파일**

- 수정: `src/kbo_pipeline/realtime_prediction.py`
- 수정: `scripts/ops/predict_2026.py`
- 수정: `tests/test_realtime_prediction.py`

**인터페이스**

- 생성: `prediction_window_is_open(now: Timestamp, game_datetime: Timestamp) -> bool`
- 생성: `feature_prior_usage_rate(row: Series | None) -> float`
- 예측 행: 배포 식별자, 평가 역할, 시작일, 라인업 상태, 피처 사전분포 사용률과 체크섬 포함
- 전달 행: `api_status`가 `success` 또는 `failed`, 실패 시 `error_type` 포함

- [ ] **1단계: 시간 경계와 감사 필드 실패 테스트 작성**

```python
def test_prediction_window_closes_at_game_start(self):
    start = pd.Timestamp("2026-07-28T18:30:00+09:00")
    self.assertTrue(prediction_window_is_open(start - pd.Timedelta("1ns"), start))
    self.assertFalse(prediction_window_is_open(start, start))
    self.assertFalse(prediction_window_is_open(start + pd.Timedelta("1s"), start))

def test_feature_prior_usage_rate_uses_source_columns_only(self):
    row = pd.Series({
        "home_sp_source": "league_prior",
        "away_sp_source": "current_season",
        "home_sp_fip": 4.0,
    })
    self.assertEqual(feature_prior_usage_rate(row), 0.5)
```

- [ ] **2단계: 실패 확인**

```bash
PYTHONPATH=src/kbo_pipeline:src:scripts/ops \
  python3 -m unittest tests.test_realtime_prediction -v
```

예상: 두 함수가 없어 실패한다.

- [ ] **3단계: 시간 게이트와 추가 전용 로그 구현**

실시간 루프는 대상 경기의 `game_datetime`을 먼저 계산한다. 현재 시각이 시작 시각 이상이면 `record_type="expired"` 한 행만 기록하고 해당 `s_no`를 종료 상태로 둔다. 전송 최종 실패 시 기존 예측을 유지하고 아래 전달 행을 추가한다.

```python
{
    "record_type": "delivery",
    "s_no": s_no,
    "api_status": "failed",
    "error_type": exc.__class__.__name__,
    **audit_context,
}
```

- [ ] **4단계: 대상 테스트 통과 확인**

실행: 2단계와 동일  
예상: 모든 테스트 통과

- [ ] **5단계: 작업 2 커밋**

```bash
git add src/kbo_pipeline/realtime_prediction.py \
  scripts/ops/predict_2026.py tests/test_realtime_prediction.py
git commit -m "fix: 경기 후 예측과 감사 로그 보강"
```

---

### 작업 3: 전향적 평가와 통계 판정

**파일**

- 수정: `scripts/ops/evaluate_prediction_log.py`
- 수정: `tests/test_prediction_log_evaluation.py`

**인터페이스**

- 수정: `prepare_evaluation_rows(prediction_log, games, *, deployment_id) -> DataFrame`
- 생성: `paired_day_block_bootstrap(rows, baseline_probability, *, iterations=2000, seed=42) -> dict[str, float]`
- 생성: `evaluate_prediction_log(rows, baseline_probability) -> DataFrame`
- 생성: `calibration_bins(rows) -> DataFrame`
- 생성: `operating_decision(metrics: Mapping[str, float]) -> str`

- [ ] **1단계: 코호트·운영 지표·통계 실패 테스트 작성**

```python
def test_only_registered_prospective_deployment_is_evaluated(self):
    result = prepare_evaluation_rows(log, games, deployment_id="deploy-a")
    self.assertEqual(result["s_no"].tolist(), [1])

def test_overall_rates_use_all_predictions(self):
    report = evaluate_prediction_log(rows, baseline_probability=0.52)
    overall = report.loc[report["model_type"].eq("all")].iloc[0]
    self.assertEqual(overall["fallback_rate"], 0.5)
    self.assertEqual(overall["api_success_rate"], 0.5)

def test_bootstrap_is_reproducible_and_delta_is_model_minus_baseline(self):
    first = paired_day_block_bootstrap(rows, 0.52, iterations=100, seed=42)
    second = paired_day_block_bootstrap(rows, 0.52, iterations=100, seed=42)
    self.assertEqual(first, second)
    self.assertLess(first["mean_log_loss_delta"], 0)
```

단일 클래스와 10경기 미만의 보정·ROC 지표가 `NaN`인지, 판정 우선순위가 데이터→모델→재보정→관찰→유지인지 각각 독립 테스트한다.

- [ ] **2단계: 실패 확인**

```bash
PYTHONPATH=src/kbo_pipeline:src:scripts/ops \
  python3 -m unittest tests.test_prediction_log_evaluation -v
```

예상: 새 필터·함수·지표가 없어 실패한다.

- [ ] **3단계: 전향적 평가 최소 구현**

예측 행은 역할·시작일·배포 식별자·경기 전 조건을 모두 검사하고 최초 행만 남긴다. 같은 `s_no`의 전달 행에 성공이 하나라도 있으면 `delivery_success=1`, 아니면 0으로 결합한다.

보고서는 전체와 모델 유형별로 주간·월간·전체 행을 생성한다. 소표본에서도 로그 손실·브라이어·정확도는 계산하고, 보정 절편·기울기와 ROC AUC만 `NaN` 처리한다.

- [ ] **4단계: 통계 판정 구현**

블록 부트스트랩은 고유 경기일을 복원 추출하고 그 날짜의 경기별 짝지은 손실 차이를 함께 재표본한다. 운영 판정은 설계 문서의 우선순위와 다음 방향을 사용한다.

```text
차이 < 0: 모델 우위
차이 > 0: 기준선 우위
```

- [ ] **5단계: 대상 테스트 통과 확인**

실행: 2단계와 동일  
예상: 모든 테스트 통과

- [ ] **6단계: 작업 3 커밋**

```bash
git add scripts/ops/evaluate_prediction_log.py \
  tests/test_prediction_log_evaluation.py
git commit -m "feat: 전향적 운영 성능 판정 추가"
```

---

### 작업 4: 읽기 API 계약 스모크 검증

**파일**

- 생성: `scripts/verify/__init__.py`
- 생성: `scripts/verify/read_api_contract.py`
- 생성: `tests/test_read_api_contract.py`
- 출력: `verification_logs/read_api_contract.json`

**인터페이스**

- 생성: `select_smoke_targets(games, lineups, rosters) -> dict[str, Any]`
- 생성: `summarize_response(endpoint: str, response: Any, required_fields: set[str]) -> dict[str, Any]`
- 실행: `python3 -m scripts.verify.read_api_contract`

- [ ] **1단계: 표본 선택과 비밀 비저장 실패 테스트 작성**

```python
def test_summary_contains_no_response_body_or_credentials(self):
    summary = summarize_response(
        "prediction/gameLineup",
        {"result_cd": 100, "list": [{"p_no": 1}]},
        {"p_no"},
    )
    self.assertNotIn("response", summary)
    self.assertNotIn("headers", summary)
    self.assertTrue(summary["required_fields_present"])
```

- [ ] **2단계: 실패 확인**

```bash
PYTHONPATH=src/kbo_pipeline:src \
  python3 -m unittest tests.test_read_api_contract -v
```

예상: 검증 모듈이 없어 실패한다.

- [ ] **3단계: 읽기 전용 스모크 실행기 구현**

로컬 CSV에서 최신 종료 경기, 해당 경기 팀·날짜, 선발 투수와 타자를 결정한다. 다섯 엔드포인트를 호출하되 결과 파일에는 호출 시각, 엔드포인트, `result_cd`, 행 수, 필수 필드 존재 여부만 쓴다.

- [ ] **4단계: 단위 테스트 통과 확인**

실행: 2단계와 동일  
예상: 모든 테스트 통과

- [ ] **5단계: 실제 읽기 API 스모크 실행**

```bash
PYTHONPATH=src/kbo_pipeline:src \
  python3 -m scripts.verify.read_api_contract
```

예상: 다섯 엔드포인트 모두 성공하고 비밀값 없는 요약 JSON이 생성된다. 계약이 다르면 이후 데이터 수집은 실행하지 않는다.

- [ ] **6단계: 작업 4 커밋**

```bash
git add scripts/verify tests/test_read_api_contract.py
git commit -m "test: Statiz 읽기 계약 스모크 추가"
```

---

### 작업 5: 검증 계획·파이프라인 문서·무시 규칙

**파일**

- 수정: `docs/remaining_verification_plan.md`
- 전면 수정: `docs/pipeline_summary.md`
- 수정: `.gitignore`

**인터페이스**

- 표준 테스트 명령: 전체 모듈 경로를 포함하여 53개 이상 통과
- 문서 경로: 현재 `scripts/`, `src/kbo_pipeline/`, `artifacts/` 구조
- 무시 규칙: `docs/pipeline_summary.md`만 예외로 두고 `docs/`의 나머지 마크다운 무시

- [ ] **1단계: 검증 계획을 현재 명령·경로·통계 정의로 수정**

계획의 루트 스크립트 경로를 모듈 실행 경로로 바꾸고 Git 저장소가 아니라는 문장을 제거한다. 전체 테스트 명령은 다음으로 고정한다.

```bash
PYTHONPATH=src/kbo_pipeline:src:scripts/model:scripts/ops \
  python3 -m unittest discover -s tests -v
```

결측률 기준은 `2026년 핵심 피처 결측률 0%`와 `전체 기간 예외 행·출처 명시`로 구분한다.

- [ ] **2단계: `pipeline_summary.md` 전면 갱신**

현재 디렉터리 구조, v9 피처, 시간 분할, CatBoost 운영 모델, 최근 10경기 대체 모델, 경기 후 예측 차단, 추가 전용 로그, 전향적 평가와 API 전송 계약을 단일 운영 문서로 작성한다.

- [ ] **3단계: 나머지 문서 무시 규칙 추가**

`.gitignore`의 기존 개별 문서 규칙을 다음 두 줄로 교체한다.

```gitignore
/docs/**/*.md
!/docs/pipeline_summary.md
```

기존 추적 문서는 인덱스에서 제거하지 않는다.

- [ ] **4단계: 문서·무시 규칙 검증**

```bash
git check-ignore docs/remaining_verification_plan.md
git check-ignore docs/pipeline_summary.md
```

예상: 첫 명령은 무시 규칙을 출력하고, 두 번째 명령은 출력 없이 종료 코드 1을 반환한다.

- [ ] **5단계: 작업 5 커밋**

```bash
git add -f docs/remaining_verification_plan.md docs/pipeline_summary.md
git add .gitignore
git commit -m "docs: 운영 파이프라인과 검증 계획 갱신"
```

---

### 작업 6: 전체 회귀 및 운영 준비 검증

**파일**

- 참조: 모든 수정 파일
- 출력: `verification_logs/configuration_check.txt`

- [ ] **1단계: 전체 테스트 실행**

```bash
PYTHONPATH=src/kbo_pipeline:src:scripts/model:scripts/ops \
  python3 -m unittest discover -s tests -v
```

예상: 53개 이상의 테스트, 실패·오류 0건

- [ ] **2단계: 구문 컴파일 확인**

```bash
python3 -m compileall -q src scripts tests
```

예상: 종료 코드 0

- [ ] **3단계: 비밀 패턴과 절대 경로 검사**

```bash
rg -n 'API_KEY\s*=\s*"[0-9a-f]{20,}|SECRET\s*=\s*"[0-9a-f]{20,}' \
  --glob '*.py' --glob '*.json' --glob '*.md' .
```

예상: 실제 인증값과 오래된 `/Desktop/test` 경로 0건

- [ ] **4단계: 배포 메타데이터와 체크섬 확인**

모델·데이터·분할 명세가 존재하고, `prospective_start_date=2026-07-28`, 기준선, 배포 식별자와 체크섬을 계산할 수 있는지 확인한다.

- [ ] **5단계: Git 변경 범위 확인**

```bash
git status --short
git diff --check
```

예상: 요청 범위 파일만 변경되고 공백 오류가 없다.

- [ ] **6단계: 최종 검증 커밋**

검증에서 추가 수정이 발생한 경우에만 관련 파일을 커밋한다. 운영 로그, 검증 로그, 데이터와 모델 산출물은 Git에 추가하지 않는다.
