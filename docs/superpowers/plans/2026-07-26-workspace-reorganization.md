# 작업공간 재구성 실행 계획

> **에이전트 작업자용:** 이 계획은 `superpowers:executing-plans`로 작업별 검증을 수행한다. 현재 디렉터리는 Git 저장소가 아니므로 커밋 대신 이동 전·후 경로 목록과 전체 테스트 결과를 보존한다.

**목표:** 루트에는 `run_pipeline.py`만 남기고, 실행 코드·공통 모듈·문서·산출물·레거시 코드를 역할별 디렉터리로 재배치한다.

**구조:** 공통 모듈은 `src/kbo_pipeline/`, 실행 진입점은 `scripts/`, 모델과 평가 산출물은 `artifacts/`, 문서는 `docs/`, 레거시 사본은 `archive/legacy/`에 둔다. 데이터와 테스트는 각각 `data/`, `tests/`에 유지한다.

**기술 구성:** Python 3, 표준 라이브러리 `pathlib`, `unittest`, 셸 `mv`

## 전역 제약

- 루트의 일반 파일은 `run_pipeline.py`만 남긴다.
- `agent/`와 루트 `.DS_Store`는 사용자 요청에 따라 삭제한다.
- 기존 v8 데이터와 모델 비교 산출물은 삭제하지 않고 보관 위치만 변경한다.
- 이동 후 `PYTHONPATH=src/kbo_pipeline python3 -m unittest discover -s tests -v`가 통과해야 한다.
- 외부 API 호출, 모델 재학습, 데이터 재수집은 수행하지 않는다.

---

### 작업 1: 이동 전 목록과 대상 디렉터리 확정

**파일:**

- 생성: `artifacts/`, `archive/legacy/`, `config/`, `docs/`, `scripts/`, `src/kbo_pipeline/`
- 삭제: `agent/`, `.DS_Store`

- [ ] 루트 파일과 디렉터리 목록을 `artifacts/evaluations/workspace_layout_before.txt`에 기록한다.
- [ ] 대상 디렉터리를 생성한다.
- [ ] `agent/`와 루트 `.DS_Store`만 삭제한다.

**검증:** `find . -maxdepth 1 -type f` 결과에서 이동 전 목록과 삭제 대상이 명확히 구분된다.

### 작업 2: 소스·스크립트·문서·산출물 재배치

**파일:**

- 이동: 공통 모듈 14개 → `src/kbo_pipeline/`
- 이동: 수집·빌드·모델·운영 CLI 13개 → `scripts/` 하위 역할별 폴더
- 이동: Markdown 문서 → `docs/`
- 이동: 모델 파일 → `artifacts/models/`
- 이동: 비교·백테스트·튜닝 CSV/JSON → `artifacts/` 하위 폴더
- 이동: `Option_A/`, `Option_B/`, `create_feature_matrix_v7.py` → `archive/legacy/`
- 이동: `.env.example` → `config/.env.example`

- [ ] 실행 중인 v9 경로와 레거시 v7 경로가 섞이지 않도록 파일명을 역할 기반 이름으로 변경한다.
- [ ] v8 원천·최종 데이터는 감사 목적상 `data/`에 그대로 둔다.
- [ ] CatBoost 학습 로그는 `artifacts/training_logs/`로 옮긴다.

**검증:** 루트의 일반 파일이 `run_pipeline.py` 하나뿐이다.

### 작업 3: 경로·실행 진입점 갱신

**파일:**

- 수정: `run_pipeline.py`
- 수정: `src/kbo_pipeline/pipeline_config.py`
- 생성: `scripts/__init__.py`
- 생성: `tests/__init__.py`
- 수정: `docs/*.md`

- [ ] `PROJECT_ROOT`가 `src/kbo_pipeline/pipeline_config.py`에서 작업공간 루트를 계산하도록 수정한다.
- [ ] `run_pipeline.py`가 `scripts.<영역>.<모듈>` 모듈을 실행하고 `PYTHONPATH`에 `src/kbo_pipeline`을 추가하도록 수정한다.
- [ ] 모델·평가·튜닝·운영 로그 출력 경로를 `artifacts/`로 변경한다.
- [ ] 문서의 루트 파일 경로를 새 디렉터리 구조로 변경한다.

**검증:** `python3 run_pipeline.py`는 API 키 누락 시 수집 첫 단계에서 명확히 중단하고, 모듈 탐색 오류를 내지 않는다.

### 작업 4: 이동 후 정적·동적 검증 및 삭제 후보 목록

**파일:**

- 생성: `docs/deletion_candidates.md`
- 생성: `artifacts/evaluations/workspace_layout_after.txt`

- [ ] 전체 Python 파일을 컴파일한다.

```bash
PYTHONPATH=src/kbo_pipeline python3 -m compileall -q src scripts tests run_pipeline.py
```

- [ ] 전체 단위 테스트를 실행한다.

```bash
PYTHONPATH=src/kbo_pipeline python3 -m unittest discover -s tests -v
```

- [ ] `run_pipeline.py`만 루트 일반 파일인지 확인한다.
- [ ] 삭제하지 않은 레거시·캐시·중복 산출물을 `docs/deletion_candidates.md`에 이유와 함께 기록한다.

**검증:** 컴파일과 테스트 실패가 0건이며, 삭제 후보는 실제로 삭제되지 않는다.
