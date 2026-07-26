# 합성 v9 예시 데이터 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 실제 Statiz 원천 데이터나 인증정보 없이 오프라인 모델 단계의 입력 구조를 재현할 수 있는 고정 시드 v9 합성 예시 데이터와 생성기를 제공한다.

**Architecture:** `examples/`에 독립 실행 가능한 생성기를 둔다. 생성기는 고정 시드 42로 2023~2026년의 시점 유효 경기 행을 만들고, 기본 출력은 Git으로 추적되는 작은 예시 CSV다. 사용자는 `--output data/final/final_training_set_v9.csv`로 실제 모델 입력 위치에 생성할 수 있다.

**Tech Stack:** Python 표준 라이브러리, NumPy, Pandas

## 전역 제약

- 실제 선수·경기·API 응답·인증정보를 예시 데이터에 포함하지 않는다.
- 난수 생성기는 `numpy.random.default_rng(42)`로 고정한다.
- 생성 데이터는 `dataset_contract.validate_final_dataset`의 시간·점수·홈/원정 피처 대칭 조건을 만족한다.
- 사용자 지시에 따라 모델·파이프라인 테스트와 API 호출은 실행하지 않는다.

---

### 작업 1: 합성 v9 데이터 생성기와 기본 CSV 추가

**Files:**
- Create: `examples/generate_synthetic_v9_data.py`
- Create: `examples/data/final_training_set_v9_sample.csv`

**Interfaces:**
- Consumes: 선택 인자 `--output` 경로
- Produces: UTF-8 BOM CSV 형식의 유효한 v9 최종 피처 데이터셋

- [ ] `numpy.random.default_rng(42)`를 사용해 2023~2026년의 경기별 사전경기 피처와 비무승부 점수를 생성한다.
- [ ] 각 행의 `feature_cutoff_datetime`을 `game_datetime`보다 이르게 설정하고, 홈·원정 피처와 차이 피처를 함께 기록한다.
- [ ] 생성기의 기본 출력 경로에 작은 CSV를 생성한다.

### 작업 2: 사용 안내와 공개 문서 연결

**Files:**
- Create: `examples/README.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: 생성기의 `--output` 인자
- Produces: 예시 CSV 재생성·모델 입력 위치 생성·범위 제한을 설명하는 공개 문서

- [ ] 고정 시드, 합성 데이터라는 점, API 수집·피처 생성 검증용이 아니라 모델 단계 예시라는 점을 명시한다.
- [ ] 기본 예시 CSV 재생성과 `data/final/final_training_set_v9.csv`로의 출력 명령을 제공한다.
- [ ] 최상위 README의 실행 섹션에서 예시 데이터 문서로 연결한다.

### 작업 3: 정적 완료 점검

**Files:**
- Verify: `examples/generate_synthetic_v9_data.py`
- Verify: `examples/data/final_training_set_v9_sample.csv`
- Verify: `examples/README.md`
- Verify: `README.md`

- [ ] 생성기 출력 대상과 문서 명령이 일치하는지 확인한다.
- [ ] 예시 CSV에 실제 API 키·로컬 절대경로가 없는지 확인한다.
- [ ] 사용자 요청에 따라 모델 학습·튜닝·백테스트·API 호출은 실행하지 않는다.
