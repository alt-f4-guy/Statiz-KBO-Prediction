# ⚾ KBO 예측 파이프라인 — 적대적 코드 리뷰 보고서

> **리뷰 일시**: 2026-07-26  
> **대상 범위**: `src/kbo_pipeline/` (15개 모듈), `scripts/` (12개 스크립트), `run_pipeline.py`  
> **리뷰 방법론**: 3개 전문 에이전트(피처 엔지니어링, 모델 학습/평가, 운영/수집)에 의한 병렬 심층 분석

---

## 📊 요약 대시보드

| 심각도 | 건수 | 영향 범위 |
|:---:|:---:|---|
| 🔴 **Critical** | **5건** | 데이터 누수, 모델 보정 왜곡, 파이프라인 크래시 |
| 🟡 **Warning** | **5건** | 통계적 오류, 데이터 무결성, 운영 불안정 |
| 🔵 **Info** | **3건** | 성능 최적화, 코드 품질 |
| 🟢 **Good** | **3건** | 잘 설계된 시점 기준 방어 |

---

## 🔴 Critical — 즉시 수정 필요

### C-1. 연도별 상수(FIP Constant) 계산 시 심각한 데이터 누수

- **파일**: `src/kbo_pipeline/feature_matrix_v9.py` L154-161 (`_build_constants`), `src/kbo_pipeline/sabermetrics.py` L20-49 (`calculate_kbo_year_constants`)
- **문제**: FIP 상수와 리그 득점 환경(`league_runs_per_pa`)을 계산할 때 **해당 시즌 전체** 경기 기록을 집계합니다. 5월 경기의 피처를 만들 때 9월까지의 기록이 포함된 최종 `fip_constant`를 사용하게 되어, 모델이 **시즌 최종 투고타저 양상을 미리 알게** 되는 치명적 Data Leakage입니다.

```python
# feature_matrix_v9.py — 미래 포함 전체 pitching을 그대로 전달
def _build_constants(pitching, batting):
    constants = calculate_kbo_year_constants(pitching_for_constants)

# sabermetrics.py — 시즌 전체 합계로 상수 계산
totals = data.groupby("year", dropna=True).sum(min_count=1)  # ← 누수!
```

- **영향**: `sp_fip`, `rp_fip`, `bat_linear` 등 핵심 피처 전반에 미래 정보 오염 → 과적합 후 운영 성능 급락
- **수정안**: `feature_cutoff_datetime` 이전 시점까지의 기록만으로 롤링 계산하거나, **전년도 시즌 최종 상수**를 사용

---

### C-2. 확률 보정(Platt Scaling) 시 L2 정규화로 인한 수학적 왜곡

- **파일**: `src/kbo_pipeline/classifier_model.py` L198, L234
- **문제**: 시그모이드 보정 및 평가지표 계산에 사용된 `LogisticRegression`이 기본값 L2 정규화를 사용합니다. 로짓에 정규화가 적용되면 기울기가 축소되어, **확률이 강제로 0.5 부근으로 모이는** 수학적 왜곡이 발생합니다.

```python
self.model = LogisticRegression(random_state=RANDOM_STATE)  # penalty=None 누락!
```

- **영향**: 보정된 예측 확률이 under-confident하게 출력, calibration slope/intercept 지표도 왜곡
- **수정안**: `LogisticRegression(penalty=None, random_state=RANDOM_STATE)`

---

### C-3. 캘리브레이션 불일치로 인한 모델 비교 기준 붕괴

- **파일**: `scripts/model/train_classifier.py` L75, `scripts/model/compare_models.py` L84
- **문제**: 교차 검증(Dev Folds)에서는 **보정 전 원시 확률**로 Log-loss를 평가하고, 최종 2026년 테스트에서는 **보정 후 확률**을 평가합니다. `compare_models.py`에서 이 둘의 차이로 `period_instability`를 계산하는데, **서로 다른 스케일의 지표를 연산**하는 논리적 오류입니다.

```python
# train_classifier.py — 보정 미적용
probability = model.predict_proba(validation[features])[:, 1]

# compare_models.py — 보정 적용된 지표와 미적용 지표의 뺄셈
candidates["period_instability"] = (
    candidates["final_log_loss"] - candidates["development_log_loss"]
).abs()
```

- **영향**: 기간 안정성 지표가 무의미한 노이즈 → **최적이 아닌 모델이 운영에 배포**될 가능성
- **수정안**: Dev Folds 내에서도 `SigmoidCalibrator`를 적용한 뒤 지표를 계산

---

### C-4. `backtest.py`의 다형성 처리 실패 — 득점 모델 선정 시 크래시

- **파일**: `scripts/model/backtest.py` L20-21
- **문제**: `best_model.joblib`에서 로드한 모델이 득점 분포 모델(`dict` 타입)일 경우, `model.predict_proba()`를 호출하며 `AttributeError`로 크래시합니다.

```python
model = joblib.load(MODEL_DIR / "best_model.joblib")
probability = model.predict_proba(final_test[model.feature_columns])[:, 1]  # dict엔 predict_proba 없음!
```

- **영향**: 득점 모델이 최적 모델로 선정되면 파이프라인 전체 중단
- **수정안**: 모델 타입 검사 분기 (`isinstance(model, dict)`) 후 득점 모델 전용 예측 로직 태우기

---

### C-5. 타임존 불일치로 실시간 예측 파이프라인 크래시

- **파일**: `scripts/ops/predict_2026.py` L267-278
- **문제**: timezone-naive Timestamp(`game_time`)와 timezone-aware Timestamp(`pd.Timestamp.now(tz="UTC")`)를 비교하면 `TypeError`가 발생합니다.

```python
game_time = pd.Timestamp(target_time["game_datetime"])       # Naive (KST 문자열)
deadline = game_time - pd.Timedelta(minutes=LINEUP_DEADLINE_MINUTES)
if not complete and pd.Timestamp.now(tz="UTC") < deadline:   # TypeError!
```

- **영향**: 실시간 예측 스크립트가 즉시 죽음
- **수정안**: `game_time`을 KST로 명시 변환 후 `now(tz="Asia/Seoul")`과 비교

---

## 🟡 Warning — 신뢰성 저하 위험

### W-1. 대체 모델(Fallback)의 진행 중 경기 결과 누수

- **파일**: `src/kbo_pipeline/fallback_recent10.py` L19, L146
- **문제**: 최근 10경기 전적을 계산할 때 `game_datetime`(경기 시작 시각) 기준으로 필터링합니다. 경기 시작 후~종료 전 시점에서 예측하면, 아직 끝나지 않은 경기의 최종 점수를 과거 데이터로 사용하는 look-ahead bias가 발생합니다.
- **수정안**: `game_end_datetime` 필드 추가 또는 `game_datetime + 4시간` 버퍼 적용

### W-2. 불펜 K/BB 비율 계산의 수학적 오류

- **파일**: `src/kbo_pipeline/feature_matrix_v9.py` L599-616 (`_bullpen_features`)
- **문제**: 볼넷 0개인 우수한 릴리프 투수의 K/BB가 `NaN → fillna(2.0)`으로 **평범하게 평가절하**됩니다.

```python
current_kbb = grouped["current_so"].fillna(0) / grouped["current_bb"].fillna(0).replace(0, np.nan)
grouped["rp_k_bb"] = current_kbb.fillna(2.0).clip(0.25, 10.0)  # BB=0이면 무조건 2.0
```

- **수정안**: 분모에 epsilon 추가(`current_bb + 0.1`) 또는 BB=0 & SO>0일 때 상한값(10.0) 부여

### W-3. `.to_numpy()` 배열 연산으로 인한 파이프라인 무결성 위험

- **파일**: `src/kbo_pipeline/feature_matrix_v9.py` L709-714 (`_bullpen_fatigue`)
- **문제**: 두 DataFrame의 행 순서가 완벽히 일치한다고 가정하고 `.to_numpy()`로 뺄셈을 강제합니다.

```python
latest[f"rp_ip_{days}d"] = latest["cum_ip"].fillna(0) - earlier["cum_ip"].fillna(0).to_numpy()
```

- **수정안**: `s_no`와 `t_code` 기준 명시적 `merge` 후 컬럼 간 연산

### W-4. CSV 원자적 쓰기 부재로 데이터 손실 위험

- **파일**: `scripts/collect/collect_lineups.py` L83-94, `scripts/collect/collect_rosters.py` L89-99
- **문제**: 기존 데이터와 새 데이터를 병합하여 통째로 `to_csv()`하는 중 프로세스 중단 시 기존 수집 데이터 전체 유실
- **수정안**: 임시 파일에 먼저 쓴 뒤 `os.replace()`로 원자적 교체

### W-5. 실시간 예측 루프 내 메모리 누수

- **파일**: `scripts/ops/predict_2026.py` L353-356
- **문제**: `log_history` DataFrame이 무한 루프에서 `pd.concat`으로 계속 누적 → 장기 운영 시 OOM

```python
log_history = pd.concat([log_history, pd.DataFrame([prediction_record])], ignore_index=True)
```

- **수정안**: 하루 단위 리로드 또는 메모리 내 불필요한 과거 레코드 해제

---

## 🔵 Info — 개선 권장

### I-1. GroupBy 집계 시 비효율적 Lambda 패턴

- **파일**: `src/kbo_pipeline/feature_matrix_v9.py` L519-520
- **문제**: `agg` 내 `lambda`로 문자열 비교 → C 벡터화 비활성화, 순수 파이썬 루프

```python
current_count=("bat_source", lambda values: values.eq("current_season").sum()),
```

- **수정안**: `groupby` 전에 Boolean 컬럼을 미리 생성 후 `sum` 집계

### I-2. API 서명 생성 시 float 직렬화 불안정

- **파일**: `src/kbo_pipeline/statiz_api.py` L79
- **문제**: `str(float_param)` 결과가 파이썬 버전에 따라 달라질 수 있어 HMAC 서명 불일치 가능
- **수정안**: 엄격한 포맷팅(`f"{val:.2f}"`) 또는 정수형 강제 변환

### I-3. 클래스 불균형 가중치의 캘리브레이션 교란

- **파일**: `src/kbo_pipeline/classifier_model.py` L166
- **문제**: `auto_class_weights="Balanced"`가 원시 확률 분포를 왜곡 → 보정 전 교차 검증에서 CatBoost가 부당하게 페널티를 받음 (C-3과 복합 작용)
- **수정안**: 확률 추정 목적에서는 클래스 가중치 밸런싱보다 순수 Log-loss 최적화가 적합

---

## 🟢 Good — 잘 설계된 부분

### G-1. As-of-date 시점 기준 방어 (game_time.py)
`game_datetime - epsilon(1μs)`을 사용하여 `feature_cutoff_datetime`을 생성하는 로직은 경기 시작 직전 시점만을 정확히 잘라내며, Data Leakage 방어 관점에서 훌륭합니다.

### G-2. 선발 투수 분리 (asof_features.py)
`p_no != starter_p_no` 조건으로 당일 선발 투수가 불펜 후보로 혼입되는 현상을 정확히 필터링합니다.

### G-3. 구장 팩터의 시점 기준 준수 (sabermetrics.py)
`cumsum() - sum()` 패턴으로 현재 경기의 득점을 안전하게 제외하고 과거 데이터만 반영하는 로직이 완벽히 구현되어 있습니다.

---

## 🎯 수정 우선순위 권장

| 순위 | 이슈 | 근거 |
|:---:|---|---|
| 1 | **C-5** 타임존 크래시 | 운영 즉시 중단 — 가장 쉽고 빠르게 수정 가능 |
| 2 | **C-2** Platt Scaling 정규화 | 한 줄 수정으로 모든 보정 결과가 정상화 |
| 3 | **C-4** backtest 다형성 | 분기 처리 추가로 파이프라인 안정성 확보 |
| 4 | **C-3** 캘리브레이션 불일치 | 모델 비교 공정성 확보 — C-2와 함께 수정 |
| 5 | **C-1** 연도별 상수 누수 | 구조 변경 필요 — 가장 영향이 크지만 작업량도 많음 |
| 6 | **W-1~W-5** | 운영 안정성 및 통계적 정확성 순차 개선 |

> **참고**: **C-1(연도별 상수 누수)**은 모델의 모든 핵심 피처에 영향을 미치므로, 수정 후 반드시 **모델 재학습 → 백테스트 재실행**이 필요합니다. 수정 전후 성능 차이가 크다면 그만큼 누수에 의존하고 있었다는 의미입니다.
