# ⚾ KBO 경기 예측 파이프라인 적대적 리뷰 (Adversarial Review)

본 문서는 현재 KBO 경기 예측 파이프라인([create_feature_matrix_v7.py](file:///Users/wiww1030/Desktop/test/create_feature_matrix_v7.py), [tune_hyperparameters.py](file:///Users/wiww1030/Desktop/test/tune_hyperparameters.py), [predict_2026.py](file:///Users/wiww1030/Desktop/test/predict_2026.py), [backtest.py](file:///Users/wiww1030/Desktop/test/backtest.py))의 **사용 변수(Features), 모델 구조 및 확률 추정 방식, 튜닝 및 백테스팅 검증 체계**에 대해 심층 분석하고 구조적 결함과 데이터 누수(Data Leakage) 요소를 파헤친 적대적 리뷰(Adversarial Review) 보고서입니다.

---

## 🎯 1. 개요 (Executive Summary)

본 파이프라인은 5종 머신러닝 모델 앙상블 및 Skellam 분포 기반의 득점 예측 구조를 갖추고 있으나, **실전 예측 환경에서의 피처 붕괴(Feature Degeneration)**, **Sabermetrics 도메인 지표의 자의적 왜곡**, **테스트 데이터 세트(2026년)에 하이퍼파라미터를 과적합(Data Leakage)시키는 평가 구조** 등 다수의 치명적인 결함이 존재합니다.

> [!CAUTION]
> **핵심 경고**: 현재 백테스트 및 Optuna 튜닝 결과에서 표시되는 높은 적중률은 2026년 테스트 데이터 세트에 대한 파라미터 과적합(Target Leakage) 및 경기 후 수집된 키 매칭으로 인한 데이터 누수(Look-ahead Bias)에 기인한 착시일 가능성이 매우 높습니다.

---

## 🥊 2. 세부 영역별 적대적 리뷰

### 2.1. 변수 생성을 포함한 피처 엔지니어링 (Feature Architecture & Data Leakage)

#### 1. 실전 배치 시 피처 붕괴 (Real-time Feature Degeneration Leakage)
* **코드 위치**: [create_feature_matrix_v7.py:L38-L52](file:///Users/wiww1030/Desktop/test/create_feature_matrix_v7.py#L38-L52), [predict_2026.py:L190-L196](file:///Users/wiww1030/Desktop/test/predict_2026.py#L190-L196)
* **문제점**:
  - `p_lookup`과 `b_lookup` 변수는 경기가 완료된 후 수집되는 일별 데이터(`player_day_processed.csv`)의 경기 키(`s_no_key`)를 기반으로 결합됩니다.
  - 그러나 실시간 라이브 예측 시점([predict_2026.py](file:///Users/wiww1030/Desktop/test/predict_2026.py))에서는 당일 경기 기록이 당연히 존재하지 않으므로, `p_lookup.get((sp_no, s_no_val), 5.2)` 조회 시 키를 찾지 못하고 **무조건 디폴트값(FIP 5.2, wRC 92)**으로 붕괴됩니다.
  - **결과**: 오프라인 백테스트에서는 경기 후 수집된 라인업/일별 데이터 키 매칭으로 높은 성능 착시(Look-ahead bias)를 보이지만, 실제 실시간 예측 시에는 리그 최고 에이스 투수도 FIP 5.2의 평범한 투수로 평가되는 치명적 오류가 발생합니다.

#### 2. 도메인 지표(Sabermetrics) 산출 방식의 자의적 왜곡
* **코드 위치**: [create_feature_matrix_v7.py:L63](file:///Users/wiww1030/Desktop/test/create_feature_matrix_v7.py#L63)
  ```python
  current_wrc = obp_std.fillna(0.280) * 333
  ```
* **문제점**:
  - **가짜 wRC(Pseudo-wRC)**: 실제 wRC/wRC+는 리그 평균 wOBA, wOBA Scale, 장타율, 출루율, 파크팩터, 리그 ERA 등을 종합한 정밀 공식을 사용해야 함에도 불구하고, 단순히 `OBP * 333`이라는 자작 임의 산식을 사용했습니다. 이는 장타력이 뛰어난 슬러거와 출루형 단타 타자의 가치를 구분하지 못합니다.
  - **FIP 상수 고정 (`+ 3.10`)**: KBO 리그의 연도별 공인구 반발력 변화나 리그 평균 ERA에 따라 매년 변동하는 C_FIP 상수를 `3.10`으로 일률 하드코딩했습니다.

#### 3. 표본 크기를 무시한 극단적인 가중치 블렌딩 (Sample Size Insufficiency)
* **코드 위치**: [create_feature_matrix_v7.py:L47, L70](file:///Users/wiww1030/Desktop/test/create_feature_matrix_v7.py#L47)
* **문제점**:
  - 투수는 5이닝(`ip / 5.0`), 타자는 10타석(`pa / 10.0`)만 소화하면 지난 시즌 성적 반영 비율이 0%가 되고 현재 시즌 성적 100%로 전환됩니다.
  - 시즌 초반 1경기(5이닝) 부진으로 FIP가 폭등한 에이스 투수나 10타석 무안타 타자의 지표가 지나치게 왜곡되어 시즌 초반 극심한 예측 노이즈를 유발합니다. (최소 50~100이닝/타석 수준의 안정화 표본 필요)

#### 4. 머신러닝의 자율 학습을 방해하는 수동 피처 주입
* **코드 위치**: [create_feature_matrix_v7.py:L130, L135](file:///Users/wiww1030/Desktop/test/create_feature_matrix_v7.py#L130#L135)
* **문제점**:
  - `W_STARTER, W_BENCH = 1.0, 0.0`으로 설정되어 벤치 타자의 성적은 곱해지면서 0이 됩니다. 이는 `rosters.csv` 수집 및 벤치 평균 연산 로직 전체가 무용지물(Dead Code)임을 의미합니다.
  - 또한 `total_diff = (batting_diff * 0.5) - (sp_fip_diff * 0.3) - (rp_fip_diff * 0.2)`와 같이 검증되지 않은 인간의 자의적 가중치(0.5, 0.3, 0.2)를 결합하여 만든 피처는 머신러닝 모델에 편향(Bias)을 강제 주입하게 됩니다.

---

### 2.2. 모델 구조 및 앙상블 체계 (Model Architecture & Ensembling)

#### 1. 포아송 회귀(Poisson Regression)의 과산포(Overdispersion) 미적용
* **코드 위치**: [tune_hyperparameters.py:L77, L102](file:///Users/wiww1030/Desktop/test/tune_hyperparameters.py#L77#L102)
* **문제점**:
  - CatBoost 및 LightGBM에서 `loss_function="Poisson"`, `objective="poisson"`을 사용하고 있습니다.
  - 포아송 분포는 **"평균 = 분산"**을 전제로 하지만, 야구 득점 데이터는 빅이닝 및 무득점 경기로 인해 **과산포(분산 >> 평균)** 현상이 뚜렷합니다. Poisson loss는 대량 득점 경기의 아웃라이어 오차에 과도하게 민감하게 반응하여 득점 기댓값을 편향시킵니다. (Negative Binomial 분포 등이 적합)

#### 2. Skellam 분포의 독립성 가정 위배
* **코드 위치**: [tune_hyperparameters.py:L51-L56](file:///Users/wiww1030/Desktop/test/tune_hyperparameters.py#L51-L56)
* **문제점**:
  - 홈팀 득점($\mu_{home}$)과 어웨이팀 득점($\mu_{away}$)이 독립적인 포아송 확률변수라고 가정하고 Skellam 분포를 적용하여 승률을 도출합니다.
  - 그러나 실제 야구 경기는 구장 파크팩터, 날씨, 연장전 규칙, 경기 흐름에 따른 승리조/패전조 불펜 투입 전략으로 인해 양 팀 득점 간 상호 연관성(Correlation)이 존재하므로 독립성 가정이 깨집니다.

#### 3. Heterogeneous 앙상블의 전처리 비일치
* **코드 위치**: [tune_hyperparameters.py:L132-L156](file:///Users/wiww1030/Desktop/test/tune_hyperparameters.py#L132-L156)
* **문제점**:
  - 트리 모델(CatBoost, LightGBM, RF)과 선형/RBF 모델(Ridge, SVR)을 혼용하면서 범주형 변수를 단순 `pd.get_dummies` 및 `StandardScaler`로 처리했습니다.
  - One-hot encoding된 0/1 이진 변수를 StandardScaler로 스케일링하는 과정에서 변수의 통계적 의미가 왜곡되어 SVR 및 Ridge 모델의 성능 반감 요소가 됩니다.

---

### 2.3. 하이퍼파라미터 튜닝 및 평가 시스템 (Validation & Tuning Leakage)

#### 1. 치명적인 테스트 세트 타깃 누수 (Overfitting to Test Set)
* **코드 위치**: [tune_hyperparameters.py:L114-L126](file:///Users/wiww1030/Desktop/test/tune_hyperparameters.py#L114-L126)
  ```python
  df_2026 = DF_GLOBAL[DF_GLOBAL['year'] == 2026].copy()
  # 2026년 데이터로 Optuna objective 평가 진행
  ```
* **문제점**:
  - Optuna 튜닝(300 trials) 시 평가 대상 검증 세트로 **최종 테스트 세트인 2026년 데이터 전체**를 사용했습니다.
  - 즉, 2026년 경기의 정확도를 가장 높게 만드는 하이퍼파라미터 및 모델 앙상블 가중치를 직접 피팅(Fitting)한 것입니다.
  - **결과**: `backtest_results_2026.csv`에서 측정된 높은 적중률은 미래 데이터에 대한 일반화 성능이 아닌 **2026년 데이터에 암기(Memorization)된 과적합 수치**에 불과합니다.

---

## 📊 3. 구조적 맹점 종합 비교표 (Adversarial Summary Table)

| 구분 | 현재 구현 방식 | 적대적 지적 (취약점) | 영향도 (Impact) |
| :--- | :--- | :--- | :--- |
| **피처 수집** | 일별 성적(`playerDay`)의 `s_no` 매칭 | 라이브 예측 시 당일 성적 부재로 피처가 기본값(FIP 5.2 등)으로 붕괴 | **CRITICAL** |
| **타자 지표** | `OBP * 333` | 단타/홈런을 구분하지 못하는 현실 부합 불가 자작 산식 | **HIGH** |
| **표본 가중치**| 5이닝 / 10타석 기준 100% 반영 | 초반 소수 표본 노이즈로 시즌 초 예측 신뢰도 급락 | **HIGH** |
| **득점 손실함수**| Poisson Regressor | 과산포(Overdispersion)를 반영하지 못해 대량 득점 오차 왜곡 | **MEDIUM** |
| **승률 변환** | Skellam 분포 | 양 팀 득점 간 상호 상관관계(구장, 투수 교체 등) 무시 | **MEDIUM** |
| **파라미터 튜닝**| 2026년 테스트 세트 대상 Optuna 튜닝 | 테스트 세트에 직접 피팅된 완벽한 Target Leakage / Overfitting | **CRITICAL** |

---

## 💡 4. 우선순위별 개선 권고안 (Actionable Remediation Plan)

1. **[CRITICAL] 피처 생성 및 라이브 예측 룩업 방식 개편**
   - 경기 당일 이전(`gameDate < current_date`)까지의 누적 스탯만 계산하여 선수-날짜 기준(`p_no`, `date`)으로 룩업 테이블 구축. 당일 경기 데이터(`playerDay`) 의존성 완전 제거.
2. **[CRITICAL] 하이퍼파라미터 튜닝 검증 체계 재설계**
   - Optuna 튜닝 시 2026년 데이터 접근 금지. 2023~2024년 데이터로 학습하고 2025년 데이터로 검증(Validation)한 뒤, 파라미터를 고정하고 2026년 데이터는 순수 백테스트(Out-of-Sample Test) 용도로만 활용.
3. **[HIGH] 정밀한 Sabermetrics 지표 도입**
   - 타자: 리그 평균 wOBA, Slugging, OBP 기반의 표준 wOBA/wRC+ 산식 적용.
   - 투수: 연도별 리그 C_FIP 상수를 동적으로 계산 및 구장 파크 팩터 반영.
   - 안정화 표본(투수 40이닝, 타자 80타석 이상) 미달 시 이전 시즌 성적 가중치를 점진적으로 이행(Bayesian Shrinkage / Regress to Mean).
4. **[MEDIUM] 승패 직접 분류(Direct Classification) 또는 과산포 회귀 도입**
   - 득점을 거쳐 Skellam 변환을 거치는 대신, 직접적인 경기 승패 binary classification(CatBoost Classifier 등) 모델 도입 후 카테고리 칼럼 스케일링 정교화.
