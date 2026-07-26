# KBO 예측 파이프라인 잔여 검증 실행 계획

> **에이전트 작업자용:** 이 계획을 실행할 때는 `superpowers:executing-plans`를 사용하여 작업별 체크포인트를 지키십시오. 실제 `prediction/savePrediction` 호출은 외부 상태를 변경하므로 검증 대상 경기와 전송 시각을 사용자가 승인한 뒤 실행하십시오.

**목표:** API 키가 다시 제공된 이후 데이터 재수집, Statiz API 전송 계약, 실시간 대체 모델, 신규 경기 전향적 성능을 검증하여 개선된 파이프라인의 운영 가능 여부를 판정한다.

**구조:** 검증은 인증·읽기 API, 현재 시즌 재수집, 데이터셋 회귀 검사, 쓰기 API, 실시간 예측, 전향적 성능 순서로 진행한다. 앞 단계가 실패하면 뒤 단계를 실행하지 않으며, 2026년 기존 경기는 공학적 회귀 검사에만 사용하고 모델 확정 이후 신규 경기만 전향적 홀드아웃으로 인정한다.

**기술 구성:** Python 3, Pandas, NumPy, scikit-learn, CatBoost, Statiz API, CSV/JSON 감사 로그, `unittest`

## 전역 제약

- 모든 시각은 `Asia/Seoul` 기준으로 기록하고 비교한다.
- API 키, 비밀키와 실제 서명 헤더를 소스 코드, 로그, CSV 및 마크다운에 저장하지 않는다.
- `prediction/savePrediction`의 `percent`는 항상 `P(Home) × 100`이다.
- API 쓰기 검증은 사용자가 승인한 경기에서만 수행한다.
- 2026년 기존 경기 결과를 추가 튜닝, 피처 선택 또는 보정 방식 변경에 사용하지 않는다.
- 모델·보정기·피처 정의를 변경하면 전향적 평가 시작일을 새로 설정한다.
- 모든 통계 재표본과 모델 실행의 시드는 42로 고정한다.
- 기존 `final_training_set_v8.csv`, `best_hyperparameters.csv`, `backtest_results_2026.csv`를 삭제하거나 덮어쓰지 않는다.
- 현재 폴더는 Git 저장소가 아니므로 검증 증거는 파일 체크섬과 실행 로그로 보존한다.

---

## 1. 검증 범위와 완료 기준

| 검증 영역 | 현재 상태 | 완료 기준 |
|---|---|---|
| 로컬 단위 테스트 | 완료 | 전체 테스트 실패 0건 |
| v9 데이터 계약 | 완료 | `s_no` 중복 0건, 미래 피처 0건, 핵심 결측률 0% |
| 인증·읽기 API | 미검증 | 5개 읽기 엔드포인트의 정상 응답과 오류 처리가 확인됨 |
| 현재 시즌 재수집 | 미검증 | 현재 시즌 스냅샷 추가, 로스터 전용 투수 포함, 실패 응답 재시도 가능 |
| 쓰기 API 계약 | 미검증 | 홈 승률 0.40 사례의 `percent=40.0`이 서버와 화면에서 확인됨 |
| 실시간 주 모델 | 미검증 | 경기 전 피처 생성, 추론, 로그, 전송 성공 |
| 최근 10경기 대체 모델 | 로컬 검증 완료·실시간 미검증 | 지정 발동 조건에서만 사용되고 전송이 중단되지 않음 |
| 전향적 성능 | 데이터 대기 | 최소 100경기 조기 점검, 300경기 운영 판정 |

---

## 작업 1. 인증 정보 폐기·재발급 및 실행 환경 확인

**대상 파일**

- 참조: `.env.example`
- 참조: `pipeline_config.py`
- 참조: `statiz_api.py`
- 생성: `verification_logs/configuration_check.txt`

**인터페이스**

- 입력: `STATIZ_API_KEY`, `STATIZ_SECRET`, `STATIZ_PTT_IDX`
- 출력: 비밀값이 제거된 설정 검사 결과

- [ ] 기존 소스 코드에 노출됐던 API 키와 비밀키를 Statiz 관리 화면에서 폐기한다.
- [ ] 신규 인증 정보를 셸 또는 비밀 저장소에만 설정한다.

```bash
export STATIZ_API_KEY="<신규 API 키>"
export STATIZ_SECRET="<신규 비밀키>"
export STATIZ_PTT_IDX="<예측 계정 식별자>"
```

- [ ] 실제 값이 파일에 기록되지 않았는지 검사한다.

```bash
rg -n 'API_KEY\s*=\s*"[0-9a-f]{20,}|SECRET\s*=\s*"[0-9a-f]{20,}' \
  --glob '*.py' --glob '*.md' .
```

**예상 결과:** 검색 결과가 0건이다.

- [ ] 인증값을 제거한 별도 셸에서 설정 오류가 네트워크 호출 전에 발생하는지 확인한다.

```bash
env -u STATIZ_API_KEY -u STATIZ_SECRET -u STATIZ_PTT_IDX \
  python3 1.collect_schedule.py
```

**예상 결과:** `STATIZ_API_KEY`, `STATIZ_SECRET` 누락 오류로 종료하며 원천 CSV 수정 시각이 바뀌지 않는다.

**중단 조건**

- 폐기되지 않은 기존 키가 남아 있거나 실제 키가 파일에서 검색되면 이후 API 검증을 중단한다.

---

## 작업 2. Statiz 읽기 API 계약 스모크 테스트

**대상 파일**

- 참조: `statiz_api.py`
- 참조: `1.collect_schedule.py`
- 참조: `2.collect_lineups.py`
- 참조: `3.collect_rosters.py`
- 참조: `4.collect_player_stats.py`
- 생성: `verification_logs/read_api_contract.json`

**검증 엔드포인트**

| 엔드포인트 | 최소 필수 응답 |
|---|---|
| `prediction/gameSchedule` | `s_no`, `gameDate`, `homeTeam`, `awayTeam` |
| `prediction/gameLineup` | `p_no`, `t_code`, `position`, `battingOrder` |
| `prediction/playerRoster` | `p_no`, `t_code`, `pj_date` 또는 요청 날짜 |
| `prediction/playerDay` | 경기별 `s_no`, `p_no`, `year_req`와 투구·타격 기록 |
| `prediction/playerSeason` | `basic.list` 또는 `deepen.list`의 선수·연도 기록 |

- [ ] 종료된 경기 1개, 당일 예정 경기 1개, 로스터 팀·날짜 1개, 투수 1명, 타자 1명을 테스트 대상으로 고정한다.
- [ ] 각 GET 요청의 HTTP 상태, `result_cd`, 필수 열 존재 여부, 응답 행 수와 호출 시각만 기록한다.
- [ ] 응답 본문 전체와 인증 헤더는 검증 로그에 저장하지 않는다.
- [ ] 존재하지 않는 `s_no`를 한 번 요청해 명확한 실패가 발생하는지 확인한다.
- [ ] 연결 제한 시간과 응답 제한 시간이 적용되는지 네트워크 차단 환경에서 확인한다.
- [ ] HTTP 429 응답은 제한된 횟수만 재시도하고 무한 재귀하지 않는지 확인한다.

**합격 기준**

- 정상 대상 5개 엔드포인트가 HTTP 200과 파싱 가능한 JSON을 반환한다.
- 필수 열 누락이 0건이다.
- 오류 요청이 `None`으로 조용히 무시되지 않고 `StatizAPIError`로 구분된다.
- 비밀값이 `verification_logs/read_api_contract.json`에 존재하지 않는다.

**중단 조건**

- 응답 열 이름이나 중첩 구조가 현재 파서와 다르면 수집을 실행하지 않고 계약 차이를 먼저 수정한다.

---

## 작업 3. 현재 시즌 스냅샷 재수집 검증

**대상 파일**

- 실행: `4.collect_player_stats.py`
- 출력: `data/raw/players/player_day_snapshots.csv`
- 출력: `data/raw/players/player_season_snapshots.csv`
- 출력: `data/processed/player_processing_report.json`

- [ ] 수집 전 두 스냅샷 파일의 행 수와 마지막 `fetched_at`을 기록한다.
- [ ] 첫 번째 수집을 실행한다.

```bash
python3 4.collect_player_stats.py
```

- [ ] 같은 날 두 번째 수집을 실행한다.

```bash
python3 4.collect_player_stats.py
```

- [ ] 현재 시즌 선수는 두 실행에서 서로 다른 `fetched_at` 스냅샷이 추가됐는지 확인한다.
- [ ] 종료 시즌의 정상 선수-연도는 두 번째 실행에서 중복 호출되지 않았는지 확인한다.
- [ ] `lineups.csv`에는 없고 `rosters.csv`에만 있는 투수 표본을 최소 5명 추출하고 스냅샷 존재 여부를 확인한다.
- [ ] API 오류 행의 `response_status`가 `error`이며 이후 실행의 재시도 대상인지 확인한다.
- [ ] 스냅샷 기본키 후보 `(p_no, year_req, fetched_at)` 중복을 검사한다.

**합격 기준**

- 현재 시즌 정상 스냅샷 증가량이 0보다 크다.
- 로스터 전용 투수 표본 수집률이 100%다.
- 실패 응답이 완료 상태로 분류된 사례가 0건이다.
- `fetched_at`이 모두 시간대 정보를 포함한다.

---

## 작업 4. 재수집 후 v2·v9 데이터 회귀 검사

**대상 파일**

- 실행: `5.process_raw_data.py`
- 실행: `create_feature_matrix_v9.py`
- 출력: `data/processed/player_day_processed_v2.csv`
- 출력: `data/processed/player_season_processed_v2.csv`
- 출력: `data/final/final_training_set_v9.csv`
- 출력: `data/final/feature_coverage_v9.csv`
- 출력: `data/reference/kbo_year_constants.csv`
- 생성: `verification_logs/data_regression_summary.csv`

- [ ] 최신 정상 스냅샷으로 v2 데이터를 다시 생성한다.

```bash
python3 5.process_raw_data.py
```

- [ ] v9 데이터와 커버리지 보고서를 다시 생성한다.

```bash
python3 create_feature_matrix_v9.py
```

- [ ] 전체 단위 테스트를 실행한다.

```bash
python3 -m unittest discover -s tests -v
```

**예상 결과:** 35개 이상 테스트, 실패 0건.

- [ ] 아래 항목을 재수집 전·후 및 연도별로 비교한다.

| 지표 | 허용 기준 |
|---|---:|
| `s_no` 중복 | 0건 |
| 선수-경기 `(p_no, s_no_key)` 중복 | 0건 |
| 선수-연도 `(p_no, year)` 중복 | 0건 |
| `feature_cutoff_datetime >= game_datetime` | 0건 |
| 핵심 수치 피처 무한대 | 0건 |
| 핵심 피처 결측률 | 0% |
| 2026년 수치 피처 고유값 | 각 연속형 핵심 피처 2개 초과 |
| 홈·원정 대응 피처 누락 | 0개 |

- [ ] 더블헤더 5개 대진을 표본 추출하여 2차전 기준시각이 당일 1차전 시작 전인지 확인한다.
- [ ] 서스펜디드 경기 표본이 있으면 기준시각이 재개 시각이 아닌 최초 시작 시각인지 확인한다.
- [ ] 불펜 후보 표본 20경기에서 당일 선발과 `GS/G >= 0.5` 선발 역할 투수가 제외됐는지 확인한다.
- [ ] 목표 경기의 실제 구원 등판자 열을 제거하거나 변경해도 v9 불펜 피처가 변하지 않는지 확인한다.

**중단 조건**

- 핵심 결측률이 0%를 초과하거나 기존 v9 대비 리그 사전분포 사용률이 5%포인트 이상 증가하면 모델 재학습을 중단하고 수집 커버리지를 조사한다.

---

## 작업 5. `savePrediction` 홈 승률 계약 검증

**대상 파일**

- 참조: `realtime_prediction.py`
- 참조: `predict_2026.py`
- 생성: `verification_logs/save_prediction_contract.csv`

**사전 승인 게이트**

- [ ] 사용자가 실제 저장이 허용된 테스트 경기 `s_no`와 검증 시각을 지정한다.
- [ ] 테스트 예측이 운영 화면에 노출될 수 있음을 확인한다.
- [ ] 서버가 테스트 예측 삭제 또는 덮어쓰기를 지원하는지 Statiz 운영 문서나 관리자에게 확인한다.

**검증 사례**

| 사례 | `P(Home)` | `predictWinTeam` | API `percent` | 화면 선택 팀 확률 |
|---|---:|---|---:|---:|
| 홈 우세 | 0.60 | 홈팀 | 60.0 | 60.0 |
| 원정 우세 | 0.40 | 원정팀 | 40.0 | 60.0 |

- [ ] API 전송 직전 payload를 비밀값 없이 기록한다.
- [ ] 원정 우세 사례도 `percent=40.0`으로 전송한다.
- [ ] 서버 응답의 HTTP 상태, `result_cd`, 저장 시각을 기록한다.
- [ ] Statiz 화면 또는 조회 API에서 저장된 값이 홈팀 승률인지 확인한다.
- [ ] 같은 payload를 두 번 전송해 서버의 중복 처리 방식을 확인한다.
- [ ] 실패 응답에서는 해당 `s_no`가 처리 완료로 기록되지 않는지 확인한다.

**합격 기준**

- 두 사례 모두 서버 저장값이 `P(Home) × 100`과 일치한다.
- 화면 선택 팀 확률과 API 홈팀 확률이 서로 다른 필드로 구분된다.
- 저장 성공 전 `processed_s_nos` 등록이 0건이다.
- 동일 경기의 성공 저장이 운영 로그에서 한 번만 처리된다.

**중단 조건**

- 서버가 `percent`를 선택 팀 확률로 해석한다면 실시간 운영을 중단하고 API 계약을 재확인한다. 임의로 확률을 뒤집지 않는다.

---

## 작업 6. 실시간 주 모델·대체 모델 종단 검증

**대상 파일**

- 실행: `predict_2026.py`
- 참조: `fallback_recent10.py`
- 출력: `prediction_log.csv`
- 생성: `verification_logs/realtime_scenarios.csv`

**시나리오**

| 번호 | 조건 | 예상 모델 | 예상 처리 |
|---:|---|---|---|
| 1 | 정상 라인업·정상 피처 | `primary` | CatBoost 홈 승률 전송 |
| 2 | 예측 마감까지 라인업 미완성 | `fallback_recent10` | 최근 10경기 확률 전송 |
| 3 | 선발 또는 핵심 피처 품질 실패 | `fallback_recent10` | `fallback_reason` 기록 |
| 4 | 주 모델 추론 예외 | `fallback_recent10` | 추론 오류 유형 기록 |
| 5 | 저장 API 일시 실패 | 기존 모델 유지 | 같은 확률을 재전송 |
| 6 | 프로세스 재시작 | 기존 불변 예측 재사용 | 확률 재계산·덮어쓰기 금지 |
| 7 | 더블헤더 2차전 | 주 모델 또는 대체 모델 | 1차전 결과 미사용 |

- [ ] 테스트 당일 모델과 데이터 체크섬을 기록한다.

```bash
shasum -a 256 \
  models/best_model.joblib \
  best_model_metadata.json \
  data/final/final_training_set_v9.csv
```

- [ ] 각 시나리오에서 `prediction` 로그가 API 호출 전에 기록되는지 확인한다.
- [ ] 성공 전송은 별도 `delivery` 로그로 추가되고 기존 행을 수정하지 않는지 확인한다.
- [ ] 대체 모델 확률이 `[0.35, 0.65]` 범위인지 확인한다.
- [ ] 한 팀의 과거 경기가 5경기 미만이면 학습 구간 리그 홈 승률을 사용하는지 확인한다.
- [ ] API 실패 재시도 사이에서 `home_win_probability`의 바이트 단위 값이 동일한지 확인한다.

**합격 기준**

- 7개 시나리오의 예상 모델·전송 정책 일치율이 100%다.
- `prediction_log.csv`의 동일 `s_no`에 최초 `prediction` 행이 하나만 존재한다.
- 경기 시작 이후 생성된 `prediction` 행이 0건이다.
- 대체 모델 발동 원인이 빈 문자열인 대체 예측이 0건이다.

---

## 작업 7. 전향적 홀드아웃 등록과 변경 동결

**대상 파일**

- 참조: `best_model_metadata.json`
- 출력: `prediction_log.csv`
- 생성: `prospective_evaluation_registry.md`
- 생성: `verification_logs/deployment_checksums.txt`

- [ ] 작업 1~6을 모두 통과한 다음 첫 운영 예측일을 `prospective_start_date`로 등록한다.
- [ ] 아래 파일의 SHA-256 체크섬을 고정한다.

```text
models/best_model.joblib
best_model_metadata.json
data/final/final_training_set_v9.csv
data/final/time_split_manifest.json
feature_matrix_v9.py
predict_2026.py
```

- [ ] 전향적 평가 기간에는 하이퍼파라미터, 보정기, 피처 정의와 확률 제한 범위를 변경하지 않는다.
- [ ] 코드 또는 모델 변경이 필요하면 새 `model_version`과 새 전향적 시작일을 발급한다.
- [ ] 2026년 기존 결과에는 `evaluation_role=engineering_regression`을 부여한다.
- [ ] 모델 확정 이후 신규 경기만 `evaluation_role=prospective_holdout`으로 분류한다.

**합격 기준**

- 모든 전향적 예측이 경기 시작 전에 기록된다.
- 예측 당시 모델·데이터 체크섬을 역추적할 수 있다.
- 재학습으로 과거 예측 확률이 변경된 사례가 0건이다.

---

## 작업 8. 전향적 성능 감시와 운영 판정

**대상 파일**

- 수정·실행: `evaluate_prediction_log.py`
- 수정: `tests/test_prediction_log_evaluation.py`
- 출력: `prediction_performance_report.csv`
- 참조: `model_comparison_results.csv`
- 참조: `fallback_recent10_metrics.csv`

- [ ] 현재 구현에 없는 보정 절편·기울기, 수신자 조작 특성 곡선 아래 면적과 피처 사전분포 사용률 집계를 추가한다.
- [ ] 단일 클래스 또는 10경기 미만의 작은 기간은 계산 불가 지표를 `NaN`으로 기록하고 보고서 생성을 중단하지 않는 테스트를 추가한다.
- [ ] 경기 시작 이후 생성된 예측 제외 테스트와 모델 유형별 분리 테스트를 실행한다.

```bash
python3 -m unittest tests.test_prediction_log_evaluation -v
```

- [ ] 경기 종료 데이터가 수집된 뒤 성능 보고서를 생성한다.

```bash
python3 evaluate_prediction_log.py
```

- [ ] 주간·월간·모델 유형별로 아래 지표를 기록한다.

| 구분 | 지표 |
|---|---|
| 확률 품질 | 로그 손실, 브라이어 점수 |
| 보정 | 보정 절편, 보정 기울기, 10구간 실제 홈 승률 |
| 판별력 | 수신자 조작 특성 곡선 아래 면적 |
| 운영 | 정확도, 대체 모델 사용률, API 저장 성공률 |
| 데이터 품질 | 선발·타선·불펜 사전분포 사용률, 라인업 완성률 |

- [ ] 100경기에서 조기 경고 검사를 수행하되 모델 채택·폐기 결론은 내리지 않는다.
- [ ] 300경기에서 첫 운영 판정을 수행한다.
- [ ] 경기일을 재표본 단위로 하는 2,000회 블록 부트스트랩을 `seed=42`로 수행한다.
- [ ] CatBoost와 고정 홈 승률 기준선의 경기별 로그 손실·브라이어 점수 차이에 대한 95% 신뢰구간을 계산한다.
- [ ] 주 모델과 실제 발동한 대체 모델의 성능을 분리한다.

**운영 판정 기준**

| 판정 | 조건 |
|---|---|
| 유지 | 300경기 이상에서 로그 손실 또는 브라이어 점수 중 하나가 기준선보다 낮고, 다른 지표가 통계적으로 명확히 악화되지 않음 |
| 추가 관찰 | 신뢰구간이 0을 포함하거나 보정 기울기가 0.8~1.2 밖이지만 확률 지표가 기준선보다 나쁘지 않음 |
| 재보정 검토 | 보정 절편 절댓값이 0.10 초과 또는 보정 기울기가 0.8~1.2 밖이며 300경기 이상 |
| 모델 재검토 | 로그 손실과 브라이어 점수가 모두 기준선보다 나쁘고 두 차이의 95% 신뢰구간 하한이 0보다 큼 |
| 데이터 파이프라인 점검 | 월간 대체 모델 사용률이 20% 초과하거나 핵심 피처 사전분포 사용률이 직전 4주보다 10%포인트 이상 증가 |

**주의:** 성능 저하와 데이터 커버리지 저하가 동시에 발생하면 모델 재학습보다 수집·피처 파이프라인을 먼저 조사한다.

---

## 작업 9. 최종 운영 승인

**필수 증거**

- [ ] `verification_logs/configuration_check.txt`
- [ ] `verification_logs/read_api_contract.json`
- [ ] `verification_logs/data_regression_summary.csv`
- [ ] `verification_logs/save_prediction_contract.csv`
- [ ] `verification_logs/realtime_scenarios.csv`
- [ ] `verification_logs/deployment_checksums.txt`
- [ ] `prediction_performance_report.csv`

**승인 체크리스트**

- [ ] 실제 인증값이 저장소와 검증 로그에 없다.
- [ ] 현재 시즌 스냅샷 재수집과 로스터 전용 투수 수집이 확인됐다.
- [ ] v9 데이터 계약과 시점 누수 회귀 검사가 통과했다.
- [ ] `savePrediction.percent`가 홈팀 승리 확률임을 실제 서버에서 확인했다.
- [ ] 주 모델과 대체 모델의 7개 실시간 시나리오가 모두 통과했다.
- [ ] 전향적 모델 버전과 체크섬이 동결됐다.
- [ ] 최소 300경기의 전향적 평가에서 유지 또는 추가 관찰 판정을 받았다.

## 최종 완료 정의

다음 조건을 모두 충족해야 잔여 검증이 완료된 것으로 판정한다.

1. 읽기·쓰기 API 계약이 실제 호출에서 확인됐다.
2. 재수집된 현재 시즌 데이터로 v9 데이터셋이 품질 기준을 통과했다.
3. 홈팀 승리 확률 전송, 성공 후 처리 완료, 동일 확률 재시도 정책이 운영 환경에서 확인됐다.
4. 최근 10경기 대체 모델이 지정 조건에서만 발동했다.
5. 모델 확정 이후 신규 비무승부 경기 300개 이상이 전향적으로 평가됐다.
6. 모든 증거 파일에서 인증 정보가 제거되고 모델·데이터 버전을 역추적할 수 있다.
