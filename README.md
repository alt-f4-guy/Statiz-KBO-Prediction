# Statiz KBO Prediction Pipeline

Statiz API를 활용하여 KBO 프로야구 데이터를 수집하고, 선수 및 팀 전력 지표를 가공하여 경기 결과를 예측하는 머신러닝 파이프라인 프로젝트입니다. 

## 프로젝트 정보
* 개발 기간: 2025.12.20 ~ 진행 중
* 주요 목적: 최근 경기 흐름과 선수별 전력 지표(FIP, wRC+ 등) 및 당일 선발 로스터 정보를 결합하여 2026시즌 KBO 경기 결과를 예측합니다.

## 기술 스택
* 언어: Python 3.14
* 주요 라이브러리: Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, Requests, python-dotenv, Rich

## 모델 설계 및 예측 방법론
본 프로젝트는 양 팀의 예상 득점(Score)을 개별적으로 예측하고 이를 통계적으로 연산하여 최종 승패 확률을 추론하는 구조를 가집니다.

### 1. 개별 득점 모델링 (Poisson Regression)
야구 경기 득점은 정수형 카운트 데이터이며 단기간 내 발생하는 사건이라는 특성을 가집니다. 이를 반영하기 위해 포아송(Poisson) 분포를 목적 함수로 하는 회귀 모델을 구성하여 홈팀 득점(homeScore)과 원정팀 득점(awayScore)을 각각 예측합니다.
* **CatBoost Regressor**: 범주형 특성(홈/원정 팀 코드) 간의 관계를 효과적으로 학습하며 Poisson 손실 함수를 사용하여 점수를 추정합니다.
* **XGBoost Regressor**: 결측치 처리와 대규모 피처 연산에 유리한 히스토그램 트리 빌더 모델로, count:poisson 목적 함수를 사용합니다.
* **LightGBM Regressor**: 리프 중심(Leaf-wise) 트리 분할 방식으로 미세한 비선형 관계를 학습하며 poisson 목적 함수로 학습을 진행합니다.
* **Ridge Regression**: 선형 제약을 통해 규제를 가함으로써 트리 모델들의 과적합을 방지하고 일반화 성능을 보완합니다.

### 2. 가중치 학습 (Sample Weight)
시간 흐름에 따른 구단 및 선수들의 최신 기량을 모델에 더욱 민감하게 반영하기 위해, 최근 경기에 더 높은 가중치를 주는 가중치(sample_weight)를 적용하여 학습을 진행합니다.

### 3. 승리 확률 산출 (Skellam 분포 활용)
앙상블 예측을 통해 얻은 홈팀의 기대 득점(lambda_home)과 원정팀의 기대 득점(lambda_away)을 바탕으로, 두 독립적인 포아송 분포 변수의 차를 설명하는 스켈람(Skellam) 분포를 정의합니다. 이를 활용하여 홈팀이 승리할 확률과 원정팀이 승리할 확률을 수학적으로 연산하여 단순 승패 이진 분류보다 높은 수준의 확률적 인사이트를 제공합니다.

## 디렉터리 구조
```text
├── 1.collect_schedule.py       # 일정 및 경기 결과 수집
├── 2.collect_lineups.py        # 선발 라인업 수집
├── 3.collect_rosters.py        # 로스터(엔트리) 정보 수집
├── 4.collect_player_stats.py   # 선수별 누적 및 일별 스탯 수집
├── 5.process_raw_data.py       # 원본 데이터 전처리 및 중간 저장
├── create_feature_matrix_v7.py # 피처 엔지니어링 및 훈련 데이터셋 생성
├── tune_hyperparameters.py     # Optuna 기반 하이퍼파라미터 최적화
├── backtest.py                 # 시계열 기반 모델 백테스트
├── predict_2026.py             # 2026 시즌 경기 결과 예측 및 추론
├── run_pipeline.py             # 수집부터 가공까지의 통합 실행 스크립트
├── requirements.txt            # 의존성 패키지 정의
├── .env.example                # 환경 변수 템플릿
└── data/
    ├── raw/                    # 수집된 원본 CSV 데이터 저장 경로
    ├── processed/              # 중간 전처리 완료된 데이터 저장 경로
    ├── final/                  # 훈련 모델에 직접 입력되는 최종 데이터 저장 경로
    └── sample/                 # 깃허브 공개용 데이터 스키마 샘플 경로
```

## 개발 및 실행 가이드

### 1. 환경 설정
필요한 패키지를 설치하고 환경 변수를 설정합니다.
```bash
pip install -r requirements.txt
```

프로젝트 루트 디렉터리에 `.env` 파일을 생성하고 아래와 같이 Statiz API 인증 정보를 기입합니다. (해당 파일은 `.gitignore`에 의해 버전 관리에서 제외됩니다.)
```env
STATIZ_API_KEY=your_api_key_here
STATIZ_SECRET=your_api_secret_here
```

### 2. 파이썬 스크립트 실행 순서

#### 데이터 수집 및 전처리 파이프라인
통합 파이프라인을 기동하여 데이터를 차례대로 수집하고 전처리합니다.
```bash
python run_pipeline.py
```
개별적으로 수행할 경우 아래 순서를 따릅니다:
1. `1.collect_schedule.py`: 경기 일정 및 결과 수집
2. `2.collect_lineups.py`: 경기별 선발 라인업 수집
3. `3.collect_rosters.py`: 구단별 1군 로스터 수집
4. `4.collect_player_stats.py`: 선수 일별/시즌별 스탯 수집
5. `5.process_raw_data.py`: 수집된 데이터의 스키마 및 결측치 전처리

#### 피처 생성 및 모델 훈련
1. **피처 매트릭스 구성**: 
   ```bash
   python create_feature_matrix_v7.py
   ```
   팀 단위 롤링 지표, 투타 하이브리드 전력 평가 지표 등을 조합하여 최종 학습용 피처셋을 빌드합니다.
   
2. **하이퍼파라미터 튜닝**: 
   ```bash
   python tune_hyperparameters.py
   ```
   Optuna를 기반으로 예측 오차를 최소화하는 CatBoost, LightGBM 등의 최적 매개변수를 탐색하고 결과를 저장합니다.

3. **백테스트 평가**:
   ```bash
   python backtest.py
   ```
   과거 데이터를 바탕으로 시계열 Rolling-window 방식으로 모델의 예측 성능을 시뮬레이션하고 평가 지표를 기록합니다.

4. **2026 시즌 예측**:
   ```bash
   python predict_2026.py
   ```
   수집 완료된 당일 데이터와 라인업을 불러와 오늘 예정된 경기 결과를 예측합니다.

## 주의 사항
* 본 프로젝트는 비공개 API인 Statiz API를 호출합니다. 승인되지 않은 환경에서 대량의 무단 호출을 지양하며, 로컬 환경에서 재현 테스트 시에는 `data/sample/` 디렉터리에 포함된 샘플 데이터를 `data/raw/` 및 하위 폴더에 복사하여 구조를 확인한 후 진행하시는 것을 권장합니다.
