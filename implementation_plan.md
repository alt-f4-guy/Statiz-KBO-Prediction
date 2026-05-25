# KBO 예측 파이프라인 코드 최적화 계획

현재 KBO 경기 예측 프로젝트의 각 스크립트를 분석한 결과, 대량의 데이터 처리 및 반복 학습 과정에서 상당한 성능 저하(병목)가 예상되는 구간들을 발견했습니다. 이들을 최적화하여 전체 파이프라인 수행 속도를 비약적으로 향상시키기 위한 구체적인 계획입니다.

## User Review Required

> [!IMPORTANT]
> **최적화 대상 우선순위 및 검증 방식 확인**
> 1. **기능적 무결성**: 모든 최적화 작업은 기존 계산 결과와 동일한 출력을 보장하는 방향으로 진행됩니다. (즉, 피처 값이나 학습 파이프라인의 결과물이 기존과 완전히 동일해야 합니다.)
> 2. **최적화 우선순위**: 
>    - 1순위: `create_feature_matrix_v7.py` (피처 매트릭스 생성 - 루프 내 슬라이싱 제거 및 벡터화)
>    - 2순위: `5.process_raw_data.py` (JSON 파싱 및 로우 단위 iteration 최적화)
>    - 3순위: `backtest.py` / `tune_hyperparameters.py` (시계열 모델 학습 병렬화 및 Optuna 프루닝 도입)

## Open Questions

> [!WARNING]
> **사용자에게 드리는 질문**
> 1. 현재 어떤 스크립트가 실행될 때 가장 오래 걸리거나 비효율적이라고 느끼시나요? (특히 불편함을 겪고 계신 파일이 있다면 말씀해 주세요.)
> 2. `backtest.py`와 `tune_hyperparameters.py`에서 매 날짜(Date)마다 모델을 완전히 처음부터 새로 학습시키는 일별 롤링 윈도우 방식을 취하고 있습니다. 날짜별 학습을 **멀티프로세싱(Parallel Processing)**으로 동시에 수행해도 괜찮을까요? 아니면 모델 학습 횟수를 조절(예: 3일 또는 1주일 단위 갱신)하는 것도 고민해 볼 수 있습니다.

---

## Proposed Changes

### Component 1: Feature Matrix Creation (`create_feature_matrix_v7.py`)

가장 심각한 $O(N \times M)$ 성능 병목 구간입니다. `merge_features_v7` 함수는 각 게임(`iterrows()`)을 돌면서 매번 전체 `lineups_df`를 필터링하고 있습니다.

#### [MODIFY] [create_feature_matrix_v7.py](file:///Users/wiww1030/Desktop/Statiz%20KBO%20Prediction%20Contest/create_feature_matrix_v7.py)
* **`merge_features_v7` 최적화**: 
  - `lineups_df`를 `s_no` 기준으로 미리 `groupby`하여 딕셔너리 혹은 인덱스 구조로 캐싱해 둡니다. 루프 내에서 데이터프레임 전체 스캔을 없애고 $O(1)$ 조회로 변경합니다.
  - `roster_lookup`을 생성할 때 `rosters_df.groupby(...)` 결과를 활용한 것처럼, 라인업 데이터도 미리 정제하여 빠른 조회가 가능하도록 구조화합니다.
* **`blend_stats` 및 `blend_wrc` 벡터화**:
  - `temp_df.apply(..., axis=1)` 대신 Pandas의 컬럼 간 연산(벡터화) 또는 매핑(`.map()`)을 사용하도록 리팩토링합니다.

---

### Component 2: Raw Data Processing (`5.process_raw_data.py`)

#### [MODIFY] [5.process_raw_data.py](file:///Users/wiww1030/Desktop/Statiz%20KBO%20Prediction%20Contest/5.process_raw_data.py)
* **JSON 파싱 및 Iteration 벡터화**:
  - `iterrows()`를 사용해 매 로우마다 `json.loads`를 호출하는 부분을 `pd.Series.apply` 또는 리스트 컴프리헨션을 이용해 C-level 루프로 처리 속도를 가속합니다.
  - 불필요한 중복 체크 및 데이터 조회 루프를 Dictionary 기반으로 단일 패스로 처리하도록 최적화합니다.

---

### Component 3: Backtesting & Hyperparameter Tuning (`backtest.py`, `tune_hyperparameters.py`)

#### [MODIFY] [backtest.py](file:///Users/wiww1030/Desktop/Statiz%20KBO%20Prediction%20Contest/backtest.py)
* **병렬 학습 도입**:
  - `joblib` 또는 Python `multiprocessing`을 사용하여 `test_dates` 루프를 여러 CPU 코어에서 병렬로 학습 및 예측하도록 변경합니다.
  - 이를 통해 멀티코어 환경에서 백테스트 수행 속도를 최대 수 배 이상 단축할 수 있습니다.

#### [MODIFY] [tune_hyperparameters.py](file:///Users/wiww1030/Desktop/Statiz%20KBO%20Prediction%20Contest/tune_hyperparameters.py)
* **Optuna 병렬 처리 및 프루닝(Pruning) 도입**:
  - Optuna 최적화 과정에서 학습 초기 성능이 저조한 시도는 조기에 중단할 수 있도록 `Optuna Pruner`를 도입합니다.
  - `study.optimize` 호출 시 `n_jobs=-1`을 주거나 적절한 워커 수를 설정하여 시도(Trial)를 병렬화합니다.

---

## Verification Plan

### Automated Tests
최적화 전후의 결과물 데이터 정합성 검증용 스크립트를 작성하여 확인합니다.
- `diff_check.py` [NEW]를 제작하여 기존 원본 데이터셋(`final_training_set_v8.csv`)과 최적화된 코드로 새로 만든 데이터셋이 소수점 이하 자리까지 정확히 일치하는지 비교 검증합니다.

### Manual Verification
- 파이프라인 수행 시간 비교 측정 및 CPU 사용량 모니터링
