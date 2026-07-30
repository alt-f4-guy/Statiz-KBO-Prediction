# 데이터 분석가 포트폴리오 README 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 현재 프로젝트의 README를 채용 담당자와 기술 면접관이 함께 읽을 수 있는 데이터 분석가 포트폴리오 문서로 개편한다.

**Architecture:** README 상단은 문제·규모·성과를 빠르게 전달하는 포트폴리오 퍼널로 구성하고, 하단은 데이터 누수 방지, 피처 설계, 시간 순서 검증, 모델 비교, 실시간 운영과 재현 근거를 제공한다. 모든 수치는 저장된 평가 산출물에서 가져오며 팀 성과와 개인 기여를 분리한다.

**Tech Stack:** Markdown, Mermaid, Python 3.14, Pandas, CatBoost, Statiz API

## Global Constraints

- 수정 대상은 `README.md` 하나로 제한한다.
- 한국어로 작성하고 성능 비교 수치는 Markdown 표로 제시한다.
- README 전체는 약 250~350줄을 목표로 한다.
- 실제 코드와 저장된 CSV·JSON으로 확인되는 사실만 사용한다.
- 2인 팀 프로젝트임을 명시하고 개인 기여 작성 항목을 남긴다.
- 장식용 배지와 외부 이미지는 추가하지 않는다.
- 기존 코드, 데이터, 모델, 테스트와 사용자 미커밋 변경은 수정하지 않는다.
- 비밀값, 개인 식별자와 공개할 수 없는 원천 데이터는 포함하지 않는다.

---

### Task 1: 포트폴리오 퍼널형 README 작성과 검증

**Files:**
- Modify: `README.md`
- Reference: `docs/superpowers/specs/2026-07-30-portfolio-readme-design.md`
- Verify: `artifacts/evaluations/model_comparison_results.csv`
- Verify: `artifacts/models/best_model_metadata.json`
- Verify: `data/final/final_training_set_v9.csv`
- Verify: `data/final/time_split_manifest.json`

**Interfaces:**
- Consumes: 승인된 설계 명세, 모델 비교 산출물, 데이터셋 규모, 시간 분할 명세, 현재 실행 모듈 경로
- Produces: 문제 정의부터 재현 방법까지 이어지는 독립형 포트폴리오 문서 `README.md`

- [ ] **Step 1: README에 사용할 핵심 수치를 원천 산출물에서 확인**

Run:

```bash
python3 -c 'import json, pandas as pd; from pathlib import Path; data=pd.read_csv("data/final/final_training_set_v9.csv"); results=pd.read_csv("artifacts/evaluations/model_comparison_results.csv"); metadata=json.loads(Path("artifacts/models/best_model_metadata.json").read_text()); split=json.loads(Path("data/final/time_split_manifest.json").read_text()); print(data.shape); print(data.groupby("year").size().to_dict()); print(results.to_string(index=False)); print(metadata["selected_model"], metadata["selection_rule"]); print(len(split["development_folds"]))'
```

Expected:

- 최종 데이터셋은 2,608행 66열이다.
- 연도별 경기 수는 2023년 720, 2024년 720, 2025년 720, 2026년 448이다.
- 운영 모델은 CatBoost다.
- 개발 폴드는 3개다.
- 모델 비교 수치는 설계 명세의 표와 일치한다.

- [ ] **Step 2: README 상단 포트폴리오 퍼널 작성**

`README.md`의 기존 내용을 교체하고 다음 순서로 작성한다.

1. `KBO 홈팀 승리 확률 예측` 제목과 엔드투엔드 프로젝트 한 문장 요약
2. `프로젝트 한눈에 보기` 표
3. `핵심 결과` 요약
4. `문제 정의`와 분석 목표
5. `핵심 분석 과제`의 문제·해결 표

상단에 다음 사실을 포함한다.

- 2인 팀 프로젝트
- 2023~2026년 2,608경기와 66개 열
- 무승부를 제외한 조건부 홈팀 승리 확률 예측
- CatBoost 직접 분류기
- 3개 순차 개발 폴드와 2026년 438개 비무승부 경기 최종 평가
- 데이터 수집부터 API 제출과 운영 로그 평가까지 구현

- [ ] **Step 3: README 기술 근거 섹션 작성**

다음 섹션을 순서대로 작성한다.

1. `데이터 구성과 분석 단위`
2. `시점 기준 피처 엔지니어링`
3. `검증 전략`
4. `모델 비교와 선택`
5. `운영 파이프라인`

포함할 기술 근거는 다음과 같다.

- 경기 시작 직전 컷오프를 기준으로 과거 정보만 결합
- 선발 FIP, 타선 OBP·SLG·ISO·BB%·K%, 불펜 FIP·ERA·K/BB와 최근 투구 부담
- 리그 사전분포와 과거 기록 축소 추정의 출처·결측 감사 열
- 2025년 확장형 순차 개발 폴드 3개와 별도 보정 구간
- 2026년 결과를 모델 선택에 사용하지 않는 최종 평가
- CatBoost, 로지스틱, 음이항, 포아송 CatBoost, 고정 홈 승률 기준선 비교
- 정확도보다 개발 로그 손실, 브라이어 점수와 보정 품질을 우선한 선택
- 라인업·피처 품질 실패 시 최근 10경기 대체 확률 사용
- API 제출 재시도, 경기 시작 후 신규 제출 차단, 배포 체크섬, 추가 전용 로그

검증과 운영 흐름은 각각 Mermaid 다이어그램으로 표현한다.

- [ ] **Step 4: 팀 역할, 재현 방법, 저장소 안내와 한계 작성**

다음 섹션을 작성한다.

1. `팀 구성과 담당 역할`
2. `재현 방법`
3. `저장소 구조`
4. `상세 문서`
5. `한계와 다음 단계`

팀 역할에는 2인 팀 프로젝트를 명시하고 다음 작성 항목을 남긴다.

- 담당 영역
- 주요 구현 및 분석
- 핵심 의사결정
- 협업 방식
- 기여 결과

재현 방법은 Python 3.14 가상환경, 주요 패키지, `config/.env.example`,
고정 시드 42 합성 데이터, `python3 run_pipeline.py`, 튜닝·백테스트·운영
로그 평가 명령을 포함한다. `run_pipeline.py`는 일일 수집·정형화·실시간
예측 흐름이며 전체 모델 학습을 자동 수행하지 않는다고 정확히 설명한다.

한계에는 시즌 초 사전분포 의존, 2026 최종 평가와 전향적 운영 평가의 차이,
API·라인업 공개 시점 제약, 실제 데이터 공개 제약을 포함한다.

- [ ] **Step 5: README 수치와 필수 섹션을 자동 대조**

Run:

```bash
python3 -c 'from pathlib import Path; import pandas as pd; text=Path("README.md").read_text(); results=pd.read_csv("artifacts/evaluations/model_comparison_results.csv"); required=["2인 팀 프로젝트","2,608","CatBoost","0.6846","0.6930","데이터 구성","검증 전략","모델 비교","운영 파이프라인","팀 구성과 담당 역할","한계와 다음 단계"]; missing=[item for item in required if item not in text]; assert not missing, missing; cat=results.loc[results["model"].eq("catboost")].iloc[0]; base=results.loc[results["model"].eq("constant_home_rate")].iloc[0]; assert f"{cat.final_log_loss:.4f}" in text; assert f"{base.final_log_loss:.4f}" in text; print("README 핵심 내용과 수치 확인 완료")'
```

Expected: `README 핵심 내용과 수치 확인 완료`

- [ ] **Step 6: README 상대 링크와 실행 경로 검증**

Run:

```bash
python3 -c 'import re; from pathlib import Path; text=Path("README.md").read_text(); links=[target for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text) if not target.startswith(("http://","https://","#"))]; missing=[target for target in links if not Path(target.split("#",1)[0]).exists()]; assert not missing, missing; print(f"상대 링크 {len(links)}개 확인 완료")'
```

Expected: 모든 상대 링크가 존재하고 종료 코드가 0이다.

Run:

```bash
rg -n 'scripts\\.(collect|build|model|ops)|config/\\.env\\.example|examples/generate_synthetic_v9_data.py' README.md
```

Expected: README의 모듈과 파일 경로가 현재 저장소 구조와 일치한다.

- [ ] **Step 7: 비밀값, 형식과 변경 범위 검증**

Run:

```bash
rg -n 'STATIZ_API_KEY=.{8,}|STATIZ_SECRET=.{8,}|245281b86d65c6edba076ff86cc7a16d|794b143fe19e' README.md
```

Expected: 출력이 없고 종료 코드는 1이다.

Run:

```bash
git diff --check
git diff -- README.md
git status --short
```

Expected:

- 공백 오류가 없다.
- 구현 diff는 `README.md`만 포함한다.
- 기존의 다른 미커밋 변경 상태는 유지된다.

- [ ] **Step 8: README만 커밋**

Run:

```bash
git add README.md
git commit -m "docs: 데이터 분석 포트폴리오 README 개편"
```

Expected: README 변경만 포함한 문서 커밋이 생성된다.
