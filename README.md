# KBO 경기 승리확률 예측 파이프라인

Statiz API와 누적 경기 데이터를 이용해 KBO 정규시즌 경기의 **홈팀 승리 확률**을 산출·전송하는 데이터 파이프라인입니다. 데이터 수집부터 시점 기준 피처 생성, 분류 모델 학습·보정, 백테스트, 실시간 예측까지 하나의 실행 흐름으로 관리합니다.

> 이 프로젝트의 API 전송 확률(`percent`)은 예측 승리 팀과 관계없이 항상 **홈팀 승리 확률 × 100**입니다.

## 주요 구성

- **시점 기준 피처 생성**: 경기 시작 전까지 이용 가능한 선수·팀 기록만 사용합니다.
- **로스터 기반 불펜 집계**: 경기 시작 시점 로스터에 등록된 투수 가운데 당일 선발 투수를 제외한 후보군으로 불펜 피처를 구성합니다.
- **확률 중심 모델 선택**: 직접 승패 분류기와 득점 분포 모델을 동일한 경기·확률 지표로 비교하고, 운영 모델을 선택합니다.
- **순차 검증과 보정**: 시간 순서 분할, 로그 손실·브라이어 점수·보정 지표를 기반으로 튜닝과 평가를 수행합니다.
- **운영 안전장치**: 선수 피처 품질이 부족하거나 주 모델 추론이 실패하면 최근 10경기 승률 기반 대체 모델을 사용합니다.
- **예측 로그 평가**: 경기 전 최초 기록 예측만으로 전향적 성능을 집계합니다.

## 파이프라인 흐름

```text
Statiz API / 누적 원천 데이터
        ↓
일정 · 라인업 · 로스터 · 선수 기록 수집
        ↓
원천 데이터 정형화
        ↓
v9 시점 기준 피처 생성
        ↓
하이퍼파라미터 튜닝 · 분류/득점 모델 학습 · 비교
        ↓
2026 백테스트 및 운영 모델 확정
        ↓
실시간 예측 · 홈팀 승률 전송 · 예측 로그 저장
```

## 디렉터리 구조

```text
.
├── run_pipeline.py             # 전체 파이프라인 진입점
├── config/
│   └── .env.example            # 환경변수 예시
├── src/kbo_pipeline/           # 공통 수집·가공·피처·모델·API 모듈
├── scripts/
│   ├── collect/                # 일정·라인업·로스터·선수 기록 수집
│   ├── build/                  # 원천 정형화와 v9 피처 생성
│   ├── model/                  # 튜닝·학습·모델 비교·백테스트
│   └── ops/                    # 실시간 예측과 운영 로그 평가
├── data/                       # 원천·정형·최종 데이터셋
├── artifacts/
│   ├── models/                 # 운영 모델과 선택 메타데이터
│   ├── tuning/                 # 하이퍼파라미터 탐색 결과
│   ├── evaluations/            # 비교·백테스트·보정 결과
│   ├── operations/             # 실시간 예측 로그
│   └── training_logs/          # 학습 로그
├── docs/                       # 설계·개선·검증 문서
└── archive/legacy/             # 이전 구현 보관용
```

## 요구 사항

- Python 3.14
- Statiz API 인증 정보

현재 저장소에는 의존성 잠금 파일이 없으므로, 아래는 코드가 사용하는 최소 패키지 설치 예시입니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn catboost optuna scipy statsmodels joblib requests rich tqdm
```

## 설정

[`config/.env.example`](config/.env.example)를 참고하여 다음 환경변수를 실행 환경에 주입합니다. 이 프로젝트는 `.env` 파일을 자동으로 읽지 않으므로, 셸·CI·비밀 관리 도구에서 직접 설정해야 합니다.

```bash
export STATIZ_API_KEY='발급받은_API_키'
export STATIZ_SECRET='발급받은_시크릿'
export STATIZ_PTT_IDX='예측_전송_식별자'
```

API 키·시크릿·개인별 식별자는 커밋하지 마십시오.

## 재현 가능한 합성 예시 데이터

실제 Statiz 원천 데이터를 공개하지 않고도 모델 입력 구조를 확인할 수 있도록, 고정 시드 42로 생성한 v9 합성 데이터셋을 제공합니다. 생성 방법과 범위는 [examples/README.md](examples/README.md)를 참고하십시오.

```bash
# Git으로 추적되는 예시 CSV를 동일한 내용으로 다시 생성
python3 examples/generate_synthetic_v9_data.py

# 로컬 모델 입력 위치에 합성 데이터를 생성
python3 examples/generate_synthetic_v9_data.py \
  --output data/final/final_training_set_v9.csv
```

이 데이터는 API 수집·원천 정형화·실제 성능 검증을 대체하지 않으며, 오프라인 모델 단계의 실행 예시만을 위한 합성 데이터입니다.

## 실행

### 전체 파이프라인

프로젝트 루트에서 실행합니다.

```bash
python3 run_pipeline.py
```

실행 순서는 수집 → 정형화 → v9 피처 생성 → 튜닝 → 분류/득점 모델 학습 → 모델 비교 → 대체 모델 평가 → 백테스트 → 실시간 예측입니다. 마지막 실시간 예측 단계는 폴링 루프이므로 지속 실행됩니다.

### 개별 모델 작업

루트 파일을 늘리지 않고 개별 스크립트를 모듈로 실행합니다. macOS·Linux 기준 예시는 다음과 같습니다.

```bash
# CatBoost 분류기 하이퍼파라미터 튜닝
PYTHONPATH=src/kbo_pipeline:src python3 -m scripts.model.tune_hyperparameters

# 운영 모델 2026 백테스트
PYTHONPATH=src/kbo_pipeline:src python3 -m scripts.model.backtest

# 예측 로그 성능 집계
PYTHONPATH=src/kbo_pipeline:src python3 -m scripts.ops.evaluate_prediction_log
```

튜닝 횟수는 환경변수로 조정할 수 있습니다.

```bash
KBO_OPTUNA_TRIALS=50 PYTHONPATH=src/kbo_pipeline:src \
  python3 -m scripts.model.tune_hyperparameters
```

## 주요 입출력

| 단계 | 입력 | 주요 출력 |
|---|---|---|
| 수집 | Statiz API | `data/raw/` |
| 정형화 | `data/raw/` | `data/processed/` |
| 피처 생성 | 원천·정형 데이터 | `data/final/final_training_set_v9.csv` |
| 튜닝 | v9 최종 데이터셋 | `artifacts/tuning/best_classifier_hyperparameters.csv` |
| 모델 학습·비교 | v9 최종 데이터셋 | `artifacts/models/best_model.joblib` |
| 백테스트 | 운영 모델·v9 최종 데이터셋 | `artifacts/evaluations/backtest_results_v9.csv` |
| 실시간 운영 | 당일 일정·라인업·로스터 | `artifacts/operations/prediction_log.csv` |

## 운영 원칙

- 학습·검증·백테스트는 시간 순서를 지키며, 2026 시즌 데이터는 최종 평가 구간으로 분리합니다.
- 피처는 경기 시작 시각을 기준으로 계산하며, 경기 후에 확정되는 기록을 같은 경기의 입력으로 사용하지 않습니다.
- API 전송 페이로드의 `percent`는 항상 홈팀 승리 확률이고, `predictWinTeam`은 해당 확률이 0.5 이상인지에 따라 결정합니다.
- 실시간 예측이 대체 모델을 사용한 경우 모델 유형과 발동 사유를 예측 로그에 기록합니다.

## 데이터와 산출물 관리

`data/`와 `artifacts/`에는 대용량 데이터, 학습 모델, API 응답 기반 산출물이 포함될 수 있습니다. 저장소 공개 시 데이터 사용 권한과 개인정보·API 정책을 확인하고, 비밀값과 불필요한 대용량 산출물은 제외하십시오.
