# 합성 v9 예시 데이터

`data/final_training_set_v9_sample.csv`는 실제 KBO 경기·선수·Statiz API 응답을 포함하지 않는 합성 데이터입니다. 난수 시드는 42로 고정되어 있어 언제 생성해도 같은 144개 경기 행을 만듭니다.

## 포함 범위

- 2023~2026년, 시즌당 36경기의 합성 경기 전 피처
- `final_training_set_v9.csv`와 같은 열 구조
- 경기 시작 시각보다 3시간 이른 `feature_cutoff_datetime`
- 튜닝·분류기 학습·백테스트 입력에 필요한 홈·원정 피처와 비무승부 점수

## 생성

프로젝트 루트에서 실행합니다.

```bash
# GitHub에 포함된 기본 예시 CSV를 재생성
python3 examples/generate_synthetic_v9_data.py

# 로컬 모델 입력 위치에 생성
python3 examples/generate_synthetic_v9_data.py \
  --output data/final/final_training_set_v9.csv
```

생성 후 개별 모델 단계를 실행할 수 있습니다.

```bash
mkdir -p artifacts/evaluations
PYTHONPATH=src/kbo_pipeline:src python3 -m scripts.model.tune_hyperparameters
PYTHONPATH=src/kbo_pipeline:src python3 -m scripts.model.train_classifier
```

## 제한 사항

- 이 데이터는 실제 KBO 성능, 선수 능력, 팀 전력, API 응답을 나타내지 않습니다.
- API 수집·원천 정형화·v9 피처 생성 자체의 정확성을 검증하는 용도가 아닙니다.
- 실서비스 모델 성능 평가와 운영 예측에는 실제 시점 기준 데이터를 사용해야 합니다.
