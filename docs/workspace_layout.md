# 작업공간 구조

`run_pipeline.py`만 프로젝트 루트의 일반 파일로 유지한다. 전체 파이프라인은 프로젝트 루트에서 다음 명령으로 실행한다.

```bash
python3 run_pipeline.py
```

| 구분 | 위치 | 내용 |
|---|---|---|
| 공통 모듈 | `src/kbo_pipeline/` | 수집·가공·피처·모델·API 공통 로직 |
| 수집 실행 | `scripts/collect/` | 일정·라인업·로스터·선수 스냅샷 수집 |
| 구축 실행 | `scripts/build/` | 원천 데이터 정형화와 v9 피처 생성 |
| 모델 실행 | `scripts/model/` | 튜닝·학습·모델 비교·백테스트 |
| 운영 실행 | `scripts/ops/` | 실시간 예측과 예측 로그 평가 |
| 설정 예시 | `config/.env.example` | 환경변수 템플릿 |
| 산출물 | `artifacts/` | 모델, 튜닝, 평가, 운영 로그, 학습 로그 |
| 문서 | `docs/` | 설계·검증·개선 계획 |
| 보존 레거시 | `archive/legacy/` | 이전 파이프라인 구현과 결과 |

개별 실행 스크립트는 프로젝트 루트에서 모듈로 실행한다. 예를 들어 v9 피처 생성은 다음과 같다.

```bash
PYTHONPATH=src/kbo_pipeline python3 -m scripts.build.create_feature_matrix_v9
```

`run_pipeline.py`는 위 `PYTHONPATH` 설정을 자식 프로세스에 자동으로 전달한다.
