# 📊 KBO 경기 예측 모델 앙상블 다양화 개선 제안서

본 문서는 현재 KBO 예측 모델에 사용되는 알고리즘들의 과도한 중복 문제를 해소하고, 데이터셋의 규모(2,580행, 22열)를 진단하여 앙상블 시너지 효과를 극대화하기 위한 두 가지 모델 재설계안을 비교 분석한 보고서입니다.

---

## 🔍 1. 현재 예측 시스템의 진단 및 한계

* **현재 모델 구조**: CatBoost + XGBoost + LightGBM + Ridge (4종 앙상블)
* **주요 문제점**:
  1. 앙상블을 구성하는 4개 모델 중 3개가 **의사결정나무 기반 Gradient Boosting 모델**로 구성되어 있습니다. 이 모델들은 특성 파악 방식이 유사하여 예측 오류의 상관계수가 높으며, 결과적으로 단순 앙상블 평균 계산 시 분산 감소(Variance Reduction) 효과가 미미합니다.
  2. 데이터셋의 전체 행 수가 **2,580개**로, 딥러닝 기반 모델을 학습시키기에는 상대적으로 데이터 규모가 협소하여 과적합(Overfitting) 발생 가능성이 매우 큽니다.

---

## ⚖️ 2. 두 가지 모델 재설계안 비교

데이터셋 크기와 알고리즘 다양성을 종합적으로 고려한 두 가지 대안을 비교 테이블로 정리했습니다.

### 🔄 앙상블 재설계 옵션 비교

| 평가 항목 | **Option A (안전 지향형: SVR 도입)** | **Option B (실험 지향형: Shallow MLP 도입)** |
| :--- | :--- | :--- |
| **최종 모델 구성** | CatBoost + LightGBM + Random Forest + **SVR (RBF)** + Ridge | CatBoost + LightGBM + Random Forest + **Shallow MLP** + Ridge |
| **비선형 모델 다변화**| SVR의 RBF 커널을 활용한 매끄러운 예측 경계 생성 | 아주 얕은 다층 퍼셉트론을 활용한 신경망식 관계 규명 |
| **과적합 위험도** | **낮음** (마크다운 소프트 마진 규칙 기반으로 고안되어 소형 데이터에 매우 강인) | **보통 ~ 높음** (정밀한 드롭아웃 및 Weight Decay 튜닝 필수) |
| **데이터 스케일 민감도**| 높음 (Feature Scaling 필수, 현재 Ridge 스케일러 재활용 가능) | 매우 높음 (신경망 특성상 입출력 분포 제어가 중요) |
| **주요 기대 효과** | 오류 상관성이 거의 없는 전통 머신러닝의 견고한 조합 완성 | 비선형 조합을 매끄러운 고차원 함수로 피팅하여 참신한 예측값 생성 |

---

## ⚙️ 3. 변경안별 모델 세부 설계 및 하이퍼파라미터 튜닝 범위

각 변경안 도입 시 [tune_hyperparameters.py](../scripts/model/tune_hyperparameters.py)에서 탐색하게 될 핵심 파라미터 범위를 지정합니다.

### 1) Option A: SVR 도입 세부 파라미터
소규모 데이터셋에서 강력한 경계선을 도출하기 위해 scikit-learn의 `SVR` 모델을 결합합니다.

```python
# SVR 모델 선언 및 탐색 파라미터 예시
# (주석: scikit-learn SVR 패키지에서 RBF 커널을 기본으로 사용합니다.)
from sklearn.svm import SVR

svr_params = {
    "C": trial.suggest_float("svr_c", 0.1, 10.0),            # 규제 강도 (낮을수록 강한 정규화)
    "epsilon": trial.suggest_float("svr_epsilon", 0.01, 0.5),  # 마진 오차 허용 폭
    "gamma": trial.suggest_categorical("svr_gamma", ["scale", "auto"]), # 커널 영향력 반경 결정 기준
    "kernel": "rbf"                                           # 비선형 매핑 커널 고정
}
```

### 2) Option B: Shallow MLP 도입 세부 파라미터
오버핏 방지를 위하여 scikit-learn의 `MLPRegressor`를 기반으로 깊이와 뉴런 개수를 억제하여 설계합니다.

```python
# Shallow MLPRegressor 선언 및 탐색 파라미터 예시
# (주석: 오버핏 방지를 위해 Hidden layer 크기를 제한하고 높은 L2 Penalty를 부과합니다.)
from sklearn.neural_network import MLPRegressor

mlp_params = {
    # 은닉층 구조: (16, 8) 또는 (32, 16) 중 선택하여 파라미터 폭주 제한
    "hidden_layer_sizes": trial.suggest_categorical("mlp_hidden", [(16, 8), (32, 16)]),
    "alpha": trial.suggest_float("mlp_alpha", 0.01, 10.0, log=True),  # 강력한 L2 Penalty 규제 가중치
    "learning_rate_init": trial.suggest_float("mlp_lr", 0.001, 0.1, log=True), # 가중치 학습 속도
    "max_iter": 500,                                         # 최대 에포크 수
    "early_stopping": True,                                  # 검증 오차 기반 조기 종료 적용
    "random_state": 42
}
```

### 3) 앙상블 블렌딩(Blending) 가중치 최적화 공통 설정
개별 모델 예측값의 가중 합산 비율을 최적화하기 위해, Optuna를 사용해 가중치 합이 1.0에 수렴하도록 튜닝을 고도화합니다.

* **최적화 탐색 가중치**:
  - `ens_cb` (CatBoost 비중): 0.3 ~ 0.8
  - `ens_lgb` (LightGBM 비중): 0.1 ~ 0.4
  - `ens_rf` (Random Forest 비중): 0.1 ~ 0.4
  - `ens_alt` (SVR 또는 MLP 비중): 0.05 ~ 0.30
  - `ens_rd` (Ridge 비중): 0.0 ~ 0.20

---

## 🚀 4. 파이프라인 코드 수정 로드맵 (Action Items)

선택된 재설계안을 작업 공간 내 소스코드 파일들에 외과수술적으로 반영하기 위해 다음과 같이 수정을 수행합니다.

### Step 1: [tune_hyperparameters.py](../scripts/model/tune_hyperparameters.py) 수정
* XGBoostRegressor 수입 및 선언 제거.
* Random Forest Regressor 및 (SVR 또는 MLP) Regressor 추가.
* Optuna `objective` 내에 모델 학습 루프 및 앙상블 가중치 블렌딩 로직 변경.
* 최종 탐색 완료 후 `best_hyperparameters.csv`에 저장되는 형식 동기화.

### Step 2: [backtest.py](../scripts/model/backtest.py) 수정
* 백업 파라미터 매핑 딕셔너리(`DEFAULT_PARAMS`)를 신규 모델 구성 규격에 맞춰 갱신.
* 시계열 검증용 Walk-forward 루프 안의 개별 모델 학습 및 앙상블 예측 구문 업데이트.
* 백테스트 진행 추이를 보여주는 Rich 라이브러리 테이블 헤더 및 데이터 매핑 동기화.

### Step 3: [predict_2026.py](../scripts/ops/predict_2026.py) 수정
* 백 watchdog 시작 전, 갱신된 신규 앙상블 4~5개 모델로 사전 학습을 수행하도록 초기화 블록 수정.
* 당일 라인업 수집 완료 즉시 동작하는 실시간 추론 부분에서 Ridge 가공 데이터와 트리용 데이터를 다변화하여 결합 연산 수행.
