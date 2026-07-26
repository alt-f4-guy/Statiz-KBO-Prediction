# KBO 예측 파이프라인 신뢰성 개선 구현 계획

> **에이전트 작업자용:** 필수 하위 스킬로 `superpowers:subagent-driven-development` 또는 `superpowers:executing-plans`를 사용하여 작업별로 구현한다. 진행 상태는 각 단계의 체크박스로 추적한다.

**목표:** 적대적 리뷰에서 확인된 실제 결함을 제거하고, 시점 누수 없는 피처 생성·공정한 모델 선택·공통 운영 추론·안전한 데이터 저장을 검증 가능한 형태로 완성한다.

**구조:** 현재 모듈 경계를 유지하면서 시점별 리그 상수, 시간 분리 보정, 공통 확률 모델 인터페이스, 원자적 파일 쓰기만 작은 단위로 추가한다. 각 작업은 실패하는 회귀 테스트를 먼저 작성하고 최소 구현으로 통과시킨다. 모델 성능 산출물은 모든 코드 수정이 끝난 뒤 한 번만 재생성한다.

**기술 스택:** Python 3.10 이상, Pandas, NumPy, scikit-learn, CatBoost, SciPy, statsmodels, joblib, 표준 라이브러리 `unittest`

## 전역 제약사항

- 모든 시간 분할과 가용성 판정은 `Asia/Seoul`을 업무 시간대로 사용하고 내부 비교는 시간대 인식형 시각으로 수행한다.
- 데이터프레임 연산은 벡터화한다. `iterrows()`와 `DataFrame.apply()`를 새로 추가하지 않는다.
- 모델과 데이터 분할의 난수 시드는 `42`로 고정한다.
- 모델 학습·선택·보정에서 2026년 최종 테스트 정답을 의사결정에 사용하지 않는다.
- 새로운 외부 패키지는 추가하지 않는다.
- 관련 없는 리팩터링, 포맷 변경, 기존 미사용 코드 삭제는 하지 않는다.
- 운영 API 전송 스크립트는 자동 검증에서 실행하지 않는다.
- 현재 작업 디렉터리는 Git 저장소가 아니므로 아래 커밋 단계는 Git 저장소에서 실행할 때만 수행한다.
- 전체 회귀 테스트 명령은 다음으로 통일한다.

```bash
PYTHONPATH=src/kbo_pipeline:scripts/model:scripts/ops:scripts/collect:scripts/processing \
python3 -m unittest discover -s tests -v
```

---

## 작업 의존성

| 작업 | 선행 작업 | 독립 검증 결과 |
|---|---|---|
| 1. 시점별 리그 상수 | 없음 | 미래 경기 변경이 과거 피처를 바꾸지 않음 |
| 2. 최종 테스트 격리 | 없음 | 최종 지표 변경이 운영 모델 선택을 바꾸지 않음 |
| 3. 공통 확률 모델 인터페이스 | 없음 | 분류·득점 모델 모두 같은 추론 계약을 만족함 |
| 4. 개발 폴드 보정 통일 | 2 | 개발 지표가 시간 분리된 보정 확률로 계산됨 |
| 5. 보정기 정규화 제거 | 4 | 보정기와 보정 진단이 무정규화 로지스틱 회귀를 사용함 |
| 6. K/BB 및 원자적 저장 | 없음 | 0볼넷 처리와 수집 파일 보존이 검증됨 |
| 7. 결과 가용시각 계약 | 1 | 완료 결과만 대체 모델과 누적 피처에 들어감 |

---

### 작업 1: 시즌 전체 상수를 시점별 리그 상수로 교체

**파일:**

- 수정: `src/kbo_pipeline/sabermetrics.py`
- 수정: `src/kbo_pipeline/feature_matrix_v9.py`
- 수정: `tests/test_sabermetrics.py`
- 수정: `tests/test_asof_features.py`

**인터페이스:**

- 입력: 경기별 `s_no`, `year`, `feature_cutoff_datetime`
- 입력: 투수 이벤트의 `year`, `event_datetime`, `IP`, `ER`, `HR`, `BB`, `HP`, `SO`
- 입력: 타자 이벤트의 `year`, `event_datetime`, `R`, `PA`
- 출력: `calculate_asof_kbo_constants(games, pitching, batting) -> pd.DataFrame`
- 출력 열: `s_no`, `year`, `league_era`, `fip_constant`, `league_runs_per_pa`, `weight_bb`, `weight_hp`, `weight_1b`, `weight_2b`, `weight_3b`, `weight_hr`
- 기존 `calculate_kbo_year_constants()`는 전년도 최종 기록을 다음 시즌 사전값으로 계산할 때만 사용한다.

- [ ] **1단계: 미래 경기 결과 불변성 회귀 테스트 작성**

`tests/test_sabermetrics.py`에 앞선 두 경기와 극단적인 미래 경기로 구성된 테스트를 추가한다.

```python
def test_asof_constants_ignore_current_and_future_events(self):
    from sabermetrics import calculate_asof_kbo_constants

    games = pd.DataFrame(
        {
            "s_no": [1, 2],
            "year": [2026, 2026],
            "feature_cutoff_datetime": pd.to_datetime(
                [
                    "2026-04-02T18:29:59+09:00",
                    "2026-04-03T18:29:59+09:00",
                ],
                utc=True,
            ),
        }
    )
    pitching = pd.DataFrame(
        {
            "year": [2026, 2026],
            "event_datetime": pd.to_datetime(
                [
                    "2026-04-01T18:30:00+09:00",
                    "2026-04-03T18:30:00+09:00",
                ],
                utc=True,
            ),
            "IP": [9.0, 9.0],
            "ER": [3.0, 99.0],
            "HR": [1.0, 20.0],
            "BB": [2.0, 30.0],
            "HP": [0.0, 10.0],
            "SO": [8.0, 0.0],
        }
    )
    batting = pd.DataFrame(
        {
            "year": [2026, 2026],
            "event_datetime": pitching["event_datetime"],
            "R": [4.0, 99.0],
            "PA": [36.0, 36.0],
        }
    )

    original = calculate_asof_kbo_constants(games, pitching, batting)
    changed_pitching = pitching.copy()
    changed_batting = batting.copy()
    changed_pitching.loc[1, ["ER", "HR", "BB"]] = [0.0, 0.0, 0.0]
    changed_batting.loc[1, "R"] = 0.0
    revised = calculate_asof_kbo_constants(
        games, changed_pitching, changed_batting
    )

    pd.testing.assert_series_equal(
        original.loc[0],
        revised.loc[0],
        check_names=False,
    )
```

- [ ] **2단계: 새 테스트가 실패하는지 확인**

```bash
PYTHONPATH=src/kbo_pipeline \
python3 -m unittest tests.test_sabermetrics.SabermetricsTests.test_asof_constants_ignore_current_and_future_events -v
```

예상 결과: `calculate_asof_kbo_constants`를 가져올 수 없어 실패한다.

- [ ] **3단계: 시각별 합계와 누적 합계를 벡터화하여 구현**

`sabermetrics.py`에 다음 형태의 함수를 추가한다. 이벤트는 먼저 `year`, `event_datetime` 단위로 합계한 뒤 누적하고, 경기별 기준시각에는 `pd.merge_asof`의 `direction="backward"` 조건으로 결합한다. 기준시각이 경기 시작 1마이크로초 전이므로 현재 경기는 자동 제외된다.

```python
def calculate_asof_kbo_constants(
    games: pd.DataFrame,
    pitching: pd.DataFrame,
    batting: pd.DataFrame,
) -> pd.DataFrame:
    """각 경기 기준시각 이전 리그 기록만으로 상수를 계산한다."""

    requests = games[["s_no", "year", "feature_cutoff_datetime"]].copy()
    requests["feature_cutoff_datetime"] = pd.to_datetime(
        requests["feature_cutoff_datetime"], errors="coerce", utc=True
    )
    requests["_input_order"] = np.arange(len(requests))

    pitching_columns = ["IP", "ER", "HR", "BB", "HP", "SO"]
    pitching_events = pitching[
        ["year", "event_datetime", *pitching_columns]
    ].copy()
    pitching_events["event_datetime"] = pd.to_datetime(
        pitching_events["event_datetime"], errors="coerce", utc=True
    )
    pitching_totals = (
        pitching_events.groupby(["year", "event_datetime"], as_index=False)[
            pitching_columns
        ]
        .sum(min_count=1)
        .sort_values(["event_datetime", "year"])
    )
    pitching_totals[pitching_columns] = pitching_totals.groupby(
        "year", sort=False
    )[pitching_columns].cumsum()

    batting_events = batting[["year", "event_datetime", "R", "PA"]].copy()
    batting_events["event_datetime"] = pd.to_datetime(
        batting_events["event_datetime"], errors="coerce", utc=True
    )
    batting_totals = (
        batting_events.groupby(["year", "event_datetime"], as_index=False)[
            ["R", "PA"]
        ]
        .sum(min_count=1)
        .sort_values(["event_datetime", "year"])
    )
    batting_totals[["R", "PA"]] = batting_totals.groupby(
        "year", sort=False
    )[["R", "PA"]].cumsum()

    left = requests.sort_values(["feature_cutoff_datetime", "year"])
    pitching_asof = pd.merge_asof(
        left,
        pitching_totals.sort_values(["event_datetime", "year"]),
        left_on="feature_cutoff_datetime",
        right_on="event_datetime",
        by="year",
        direction="backward",
        allow_exact_matches=False,
    )
    batting_asof = pd.merge_asof(
        left,
        batting_totals.sort_values(["event_datetime", "year"]),
        left_on="feature_cutoff_datetime",
        right_on="event_datetime",
        by="year",
        direction="backward",
        allow_exact_matches=False,
    )

    valid_ip = pitching_asof["IP"].replace(0, np.nan)
    league_era = pitching_asof["ER"] * 9 / valid_ip
    fip_component = (
        13 * pitching_asof["HR"]
        + 3 * (pitching_asof["BB"] + pitching_asof["HP"])
        - 2 * pitching_asof["SO"]
    ) / valid_ip
    runs_per_pa = batting_asof["R"] / batting_asof["PA"].replace(0, np.nan)
    scale = (runs_per_pa / REFERENCE_RUNS_PER_PA).clip(0.75, 1.25)

    result = pitching_asof[["s_no", "year", "_input_order"]].copy()
    result["league_era"] = league_era.to_numpy()
    result["fip_constant"] = (league_era - fip_component).to_numpy()
    result["league_runs_per_pa"] = runs_per_pa.to_numpy()
    for name, base_weight in BASE_LINEAR_WEIGHTS.items():
        result[name] = scale.to_numpy() * base_weight
    return (
        result.sort_values("_input_order")
        .drop(columns="_input_order")
        .reset_index(drop=True)
    )
```

구현 시 첫 경기처럼 리그 누적 기록이 없는 행은 전년도 최종 상수로 보완한다. 전년도 상수도 없는 최초 연도만 기존 명시적 리그 사전값을 사용한다.

- [ ] **4단계: 피처 생성기를 전년도 상수와 시점별 상수로 분리**

`feature_matrix_v9.py`에서 `_build_constants()` 하나가 두 역할을 하지 않도록 다음 계약으로 분리한다.

```python
def _build_prior_constants(
    pitching: pd.DataFrame,
    batting: pd.DataFrame,
) -> pd.DataFrame:
    constants = calculate_kbo_year_constants(pitching)
    return add_batting_environment(constants, batting)


def _build_asof_constants(
    games: pd.DataFrame,
    pitching: pd.DataFrame,
    batting: pd.DataFrame,
    prior_constants: pd.DataFrame,
) -> pd.DataFrame:
    current = calculate_asof_kbo_constants(games, pitching, batting)
    previous = prior_constants.copy()
    previous["year"] = previous["year"] + 1
    fallback_columns = [
        "league_era",
        "fip_constant",
        "league_runs_per_pa",
        *BASE_LINEAR_WEIGHTS,
    ]
    current = current.merge(
        previous[["year", *fallback_columns]],
        on="year",
        how="left",
        suffixes=("", "_prior"),
    )
    for column in fallback_columns:
        current[column] = current[column].fillna(current[f"{column}_prior"])
    return current.drop(
        columns=[f"{column}_prior" for column in fallback_columns]
    )
```

`_prior_pitcher_table()`, `_prior_batter_table()`, `_league_batting_priors()`에는 연도별 전년도 최종 상수만 전달한다. `_pitcher_shrunk_features()`, `_batter_features()`, `_bullpen_features()`에는 `s_no`별 시점 상수를 전달하고 `year` 단독이 아니라 `["s_no", "year"]`로 결합한다.

- [ ] **5단계: 최종 피처 행 불변성 테스트 추가**

`tests/test_asof_features.py`에 미래 이벤트의 득점·투구 기록을 바꿔도 그 이전 경기의 `sp_fip`, `rp_fip`, `bat_linear`이 동일한지 검증하는 테스트를 추가한다.

- [ ] **6단계: 작업 1 테스트와 전체 테스트 실행**

```bash
PYTHONPATH=src/kbo_pipeline \
python3 -m unittest tests.test_sabermetrics tests.test_asof_features -v

PYTHONPATH=src/kbo_pipeline:scripts/model:scripts/ops:scripts/collect:scripts/processing \
python3 -m unittest discover -s tests -v
```

예상 결과: 모든 테스트가 통과하고 미래 이벤트 변경 전후의 과거 피처가 동일하다.

- [ ] **7단계: Git 저장소인 경우 작업 1 커밋**

```bash
git add src/kbo_pipeline/sabermetrics.py \
  src/kbo_pipeline/feature_matrix_v9.py \
  tests/test_sabermetrics.py \
  tests/test_asof_features.py
git commit -m "fix: 경기 시점별 리그 상수로 미래 정보 누수 제거"
```

---

### 작업 2: 2026년 최종 테스트를 모델 선택에서 완전히 격리

**파일:**

- 수정: `scripts/model/compare_models.py`
- 수정: `tests/test_model_comparison.py`
- 수정: `README.md`

**인터페이스:**

- `select_operating_model(summary: pd.DataFrame) -> str`
- 모델 선택 입력은 `development_log_loss`, `development_brier_score`, 개발 보정 거리만 사용한다.
- `final_*` 열은 선택 완료 후 보고용으로만 유지한다.

- [ ] **1단계: 최종 지표 불변성 테스트 작성**

`tests/test_model_comparison.py`에 최종 지표를 뒤집어도 개발 지표가 우수한 모델이 선택되는 테스트를 추가한다.

```python
def test_final_test_metrics_do_not_select_operating_model(self):
    from compare_models import select_operating_model

    summary = pd.DataFrame(
        {
            "model": ["classifier", "score"],
            "family": ["direct_classifier", "score_distribution"],
            "development_log_loss": [0.66, 0.68],
            "development_brier_score": [0.23, 0.24],
            "development_calibration_intercept": [0.02, 0.01],
            "development_calibration_slope": [0.98, 1.01],
            "final_log_loss": [9.0, 0.01],
            "final_brier_score": [0.90, 0.01],
            "final_calibration_intercept": [8.0, 0.0],
            "final_calibration_slope": [8.0, 1.0],
        }
    )

    selected = select_operating_model(summary)

    self.assertEqual(selected, "classifier")
```

- [ ] **2단계: 새 테스트가 기존 최종 지표 의존성 때문에 실패하는지 확인**

```bash
PYTHONPATH=src/kbo_pipeline:scripts/model \
python3 -m unittest tests.test_model_comparison.ModelComparisonTests.test_final_test_metrics_do_not_select_operating_model -v
```

예상 결과: 기존 함수가 최종 보정 거리와 최종 로그 손실을 사용하여 `score`를 선택하므로 실패한다.

- [ ] **3단계: 선택 규칙을 개발 지표만 사용하도록 축소**

`select_operating_model()`은 각 모델 계열에서 개발 로그 손실과 브라이어 점수가 가장 좋은 하나를 고른 뒤, 다음 순서로 최종 후보를 선택한다.

```python
def select_operating_model(summary: pd.DataFrame) -> str:
    """개발 구간 지표만으로 운영 모델을 선택한다."""

    candidates = (
        summary.assign(
            development_calibration_distance=(
                summary["development_calibration_intercept"].abs()
                + (summary["development_calibration_slope"] - 1).abs()
            )
        )
        .sort_values(
            [
                "development_log_loss",
                "development_brier_score",
                "development_calibration_distance",
                "model",
            ]
        )
        .reset_index(drop=True)
    )
    return str(candidates.loc[0, "model"])
```

`_model_summary()`는 개발 폴드의 보정 절편과 기울기도 평균하여 `development_calibration_intercept`, `development_calibration_slope`로 반환한다.

기존 `test_classifier_is_selected_when_probability_metrics_win_both_periods`는 `test_classifier_is_selected_when_development_metrics_win`으로 이름을 바꾸고 입력에 다음 개발 보정 열을 추가한다. 최종 지표가 선택 근거라는 기존 주석과 단언은 제거한다.

```python
"development_calibration_intercept": [0.02, 0.01],
"development_calibration_slope": [0.98, 1.01],
```

- [ ] **4단계: 메타데이터와 문서의 의미 수정**

`best_model_metadata.json`을 생성하는 코드의 `selection_rule`은 다음 문구로 변경한다.

```python
"selection_rule": (
    "개발 폴드의 로그 손실, 브라이어 점수, 보정 거리 순서; "
    "2026 최종 테스트 지표는 선택에 사용하지 않음"
)
```

`README.md`의 “2026 백테스트 및 운영 모델 확정” 표현을 “개발 지표로 운영 모델 확정 후 2026 최종 평가”로 바꾼다.

- [ ] **5단계: 모델 선택 테스트 실행**

```bash
PYTHONPATH=src/kbo_pipeline:scripts/model \
python3 -m unittest tests.test_model_comparison -v
```

예상 결과: 최종 지표를 극단적으로 변경해도 선택 결과가 변하지 않는다.

- [ ] **6단계: Git 저장소인 경우 작업 2 커밋**

```bash
git add scripts/model/compare_models.py tests/test_model_comparison.py README.md
git commit -m "fix: 최종 테스트를 운영 모델 선택에서 격리"
```

---

### 작업 3: 분류·득점 모델에 공통 확률 추론 인터페이스 제공

**파일:**

- 생성: `src/kbo_pipeline/prediction_models.py`
- 수정: `scripts/model/train_score_models.py`
- 수정: `scripts/model/backtest.py`
- 수정: `scripts/ops/predict_2026.py`
- 생성: `tests/test_prediction_models.py`

**인터페이스:**

- 공통 속성: `feature_columns: list[str]`
- 공통 메서드: `predict_proba(data: pd.DataFrame) -> np.ndarray`
- 출력 배열: `(행 수, 2)`이며 첫 열은 원정 승리 확률, 둘째 열은 홈 승리 확률
- 새 클래스: `CalibratedScoreProbabilityModel`

- [ ] **1단계: 득점 모델 공통 계약 테스트 작성**

`tests/test_prediction_models.py`에 가벼운 가짜 득점 모델을 사용한 테스트를 작성한다.

```python
import unittest

import numpy as np
import pandas as pd


class _ScoreModelStub:
    def predict(self, data):
        rows = len(data)
        return np.full(rows, 4.0), np.full(rows, 3.0)


class _CalibratorStub:
    def predict(self, probability):
        return np.asarray(probability, dtype=float)


class PredictionModelTests(unittest.TestCase):
    def test_score_model_exposes_binary_predict_proba_contract(self):
        from prediction_models import CalibratedScoreProbabilityModel

        model = CalibratedScoreProbabilityModel(
            score_model=_ScoreModelStub(),
            calibrator=_CalibratorStub(),
            feature_columns=["feature"],
        )
        probability = model.predict_proba(
            pd.DataFrame({"feature": [1.0, 2.0]})
        )

        self.assertEqual(probability.shape, (2, 2))
        np.testing.assert_allclose(probability.sum(axis=1), 1.0)
        self.assertTrue((probability[:, 1] > 0.5).all())
```

- [ ] **2단계: 새 모듈이 없어 테스트가 실패하는지 확인**

```bash
PYTHONPATH=src/kbo_pipeline \
python3 -m unittest tests.test_prediction_models -v
```

예상 결과: `prediction_models`를 가져올 수 없어 실패한다.

- [ ] **3단계: 최소 공통 래퍼 구현**

`src/kbo_pipeline/prediction_models.py`에 다음 클래스를 추가한다.

```python
"""서로 다른 모델 계열을 동일한 이진 확률 계약으로 제공한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from score_models import conditional_skellam_home_probability


@dataclass
class CalibratedScoreProbabilityModel:
    score_model: Any
    calibrator: Any
    feature_columns: list[str]

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        frame = data[self.feature_columns]
        home_mean, away_mean = self.score_model.predict(frame)
        raw = conditional_skellam_home_probability(home_mean, away_mean)
        calibrated = self.calibrator.predict(raw)
        return np.column_stack([1 - calibrated, calibrated])
```

- [ ] **4단계: 득점 모델 저장 형식을 딕셔너리에서 래퍼로 변경**

`train_score_models.py`의 `fitted_models[kind]`에는 딕셔너리 대신 다음 객체를 저장한다.

```python
fitted_models[kind] = CalibratedScoreProbabilityModel(
    score_model=model,
    calibrator=calibrator,
    feature_columns=features,
)
```

- [ ] **5단계: 백테스트와 실시간 예측의 모델 계열 분기 제거**

두 소비자는 현재 분류 모델과 동일하게 아래 계약만 사용한다.

```python
model = joblib.load(MODEL_DIR / "best_model.joblib")
features = list(model.feature_columns)
probability = model.predict_proba(frame[features])[:, 1]
```

딕셔너리 타입 검사나 모델명별 조건 분기는 추가하지 않는다. 기존 `best_score_model.joblib`은 호환되지 않으므로 모델 재학습으로 교체한다.

- [ ] **6단계: joblib 왕복 직렬화 테스트 추가**

임시 디렉터리에 `CalibratedScoreProbabilityModel`을 저장하고 다시 불러온 뒤 동일한 확률이 반환되는지 검증한다.

- [ ] **7단계: 작업 3 테스트와 관련 운영 테스트 실행**

```bash
PYTHONPATH=src/kbo_pipeline:scripts/model:scripts/ops \
python3 -m unittest \
  tests.test_prediction_models \
  tests.test_realtime_prediction \
  tests.test_model_comparison -v
```

예상 결과: 분류 모델과 득점 모델 래퍼가 모두 동일한 확률 계약을 만족한다.

- [ ] **8단계: Git 저장소인 경우 작업 3 커밋**

```bash
git add src/kbo_pipeline/prediction_models.py \
  scripts/model/train_score_models.py \
  scripts/model/backtest.py \
  scripts/ops/predict_2026.py \
  tests/test_prediction_models.py
git commit -m "fix: 모델 계열별 추론을 공통 확률 인터페이스로 통합"
```

---

### 작업 4: 개발 폴드에도 시간 분리된 확률 보정 적용

**파일:**

- 수정: `src/kbo_pipeline/time_splits.py`
- 수정: `scripts/model/train_classifier.py`
- 수정: `scripts/model/train_score_models.py`
- 수정: `tests/test_time_splits.py`
- 수정: `tests/test_classifier.py`

**인터페이스:**

- 새 함수: `split_calibration_tail(frame: pd.DataFrame, fraction: float = 0.20) -> tuple[list[int], list[int]]`
- 반환값: 시간상 앞선 기본 학습 경기 ID와 뒤따르는 보정 경기 ID
- 개발 폴드 검증 확률은 기본 모델 학습, 보정기 학습, 검증 순서의 서로 겹치지 않는 세 구간으로 생성한다.

- [ ] **1단계: 학습·보정 구간 순서 테스트 작성**

`tests/test_time_splits.py`에 날짜가 섞여 입력되어도 같은 날짜를 나누지 않고 보정 구간이 뒤에 오는지 검증한다.

```python
def test_calibration_tail_is_disjoint_and_strictly_later(self):
    from time_splits import split_calibration_tail

    frame = pd.DataFrame(
        {
            "s_no": [1, 2, 3, 4, 5, 6],
            "game_datetime": pd.to_datetime(
                [
                    "2025-04-01T18:30:00+09:00",
                    "2025-04-01T18:30:00+09:00",
                    "2025-04-02T18:30:00+09:00",
                    "2025-04-03T18:30:00+09:00",
                    "2025-04-04T18:30:00+09:00",
                    "2025-04-04T18:30:00+09:00",
                ],
                utc=True,
            ),
        }
    )

    train_ids, calibration_ids = split_calibration_tail(frame, 0.34)

    self.assertTrue(set(train_ids).isdisjoint(calibration_ids))
    train_end = frame.loc[frame["s_no"].isin(train_ids), "game_datetime"].max()
    calibration_start = frame.loc[
        frame["s_no"].isin(calibration_ids), "game_datetime"
    ].min()
    self.assertLess(train_end, calibration_start)
```

- [ ] **2단계: 새 분할 함수가 없어 테스트가 실패하는지 확인**

```bash
PYTHONPATH=src/kbo_pipeline \
python3 -m unittest tests.test_time_splits.TimeSplitTests.test_calibration_tail_is_disjoint_and_strictly_later -v
```

- [ ] **3단계: 날짜 단위 보정 꼬리 분할 구현**

```python
def split_calibration_tail(
    frame: pd.DataFrame,
    fraction: float = 0.20,
) -> tuple[list[int], list[int]]:
    """학습 프레임의 마지막 날짜 묶음을 보정 전용으로 분리한다."""

    if not 0 < fraction < 1:
        raise ValueError("fraction은 0과 1 사이여야 합니다.")
    data = frame[["s_no", "game_datetime"]].copy()
    data["game_date"] = _seoul_dates(data["game_datetime"])
    dates = np.array(sorted(data["game_date"].unique()))
    calibration_days = max(1, math.ceil(len(dates) * fraction))
    if calibration_days >= len(dates):
        raise ValueError("기본 학습과 보정에 각각 최소 한 경기일이 필요합니다.")
    calibration_dates = set(dates[-calibration_days:])
    calibration = data.loc[data["game_date"].isin(calibration_dates)]
    train = data.loc[~data["game_date"].isin(calibration_dates)]
    return (
        train["s_no"].astype("int64").tolist(),
        calibration["s_no"].astype("int64").tolist(),
    )
```

- [ ] **4단계: 분류기 개발 폴드의 보정 절차 수정**

각 개발 폴드에서 기존 `train`을 다시 기본 학습·보정으로 나눈다.

```python
base_ids, calibrator_ids = split_calibration_tail(train)
base_train = _rows_for_ids(frame, base_ids)
fold_calibration = _rows_for_ids(frame, calibrator_ids)
model = build_classifier(
    kind,
    features,
    catboost_params if kind == "catboost" else None,
)
model.fit(base_train[features], base_train["target_home_win"])
calibration_probability = model.predict_proba(
    fold_calibration[features]
)[:, 1]
calibrator = SigmoidCalibrator().fit(
    calibration_probability,
    fold_calibration["target_home_win"],
)
raw_validation = model.predict_proba(validation[features])[:, 1]
probability = calibrator.predict(raw_validation)
```

보정 구간과 검증 구간이 한 클래스만 포함하면 조용히 재사용하지 말고 분할 계약 오류를 발생시킨다.

- [ ] **5단계: 득점 모델에도 동일한 세 구간 절차 적용**

기본 득점 모델은 `base_train`으로 적합하고, `fold_calibration`의 조건부 Skellam 확률로 보정기를 적합한 뒤, 검증 구간 확률만 평가한다.

- [ ] **6단계: 보정기가 검증 정답을 보지 않는 테스트 추가**

가짜 모델과 보정기를 사용해 보정기의 `fit` 대상 ID가 검증 ID와 겹치지 않는지 검증한다. 테스트를 위해 반복 본문을 `_evaluate_classifier_fold()`와 `_evaluate_score_fold()`의 작은 함수로 추출하되, 그 외 학습 구조는 바꾸지 않는다.

- [ ] **7단계: 시간 분할·분류·득점 테스트 실행**

```bash
PYTHONPATH=src/kbo_pipeline:scripts/model \
python3 -m unittest \
  tests.test_time_splits \
  tests.test_classifier \
  tests.test_score_models -v
```

예상 결과: 각 개발 지표가 기본 학습·보정·검증의 순서를 지키며 계산된다.

- [ ] **8단계: Git 저장소인 경우 작업 4 커밋**

```bash
git add src/kbo_pipeline/time_splits.py \
  scripts/model/train_classifier.py \
  scripts/model/train_score_models.py \
  tests/test_time_splits.py \
  tests/test_classifier.py
git commit -m "fix: 개발 폴드에 시간 분리 확률 보정 적용"
```

---

### 작업 5: 보정기와 보정 진단에서 L2 정규화 제거

**파일:**

- 수정: `src/kbo_pipeline/classifier_model.py`
- 수정: `tests/test_classifier.py`

**인터페이스:**

- `SigmoidCalibrator`는 `LogisticRegression(C=np.inf, random_state=42)`를 사용한다.
- `probability_metrics()`의 보정 절편·기울기도 같은 무정규화 설정을 사용한다.
- scikit-learn 1.8에서 폐기 예정인 `penalty=None`은 사용하지 않는다.

- [ ] **1단계: 무정규화 설정 회귀 테스트 작성**

```python
def test_sigmoid_calibrator_uses_unregularized_logistic_regression(self):
    from classifier_model import SigmoidCalibrator

    calibrator = SigmoidCalibrator()

    self.assertTrue(np.isinf(calibrator.model.C))
    self.assertEqual(calibrator.model.random_state, 42)
```

- [ ] **2단계: 기본 `C=1.0` 때문에 테스트가 실패하는지 확인**

```bash
PYTHONPATH=src/kbo_pipeline \
python3 -m unittest tests.test_classifier.ClassifierTests.test_sigmoid_calibrator_uses_unregularized_logistic_regression -v
```

- [ ] **3단계: 보정기와 지표 계산을 같은 생성 함수로 통일**

중복 설정을 막기 위해 다음 비공개 생성 함수를 사용한다.

```python
def _unregularized_logistic() -> LogisticRegression:
    return LogisticRegression(C=np.inf, random_state=RANDOM_STATE)
```

`SigmoidCalibrator.__init__()`과 `probability_metrics()` 모두 이 함수를 호출한다.

- [ ] **4단계: 알려진 합성 확률로 보정 기울기 테스트 추가**

충분한 표본의 합성 로짓과 고정 타깃을 넣고, `probability_metrics()`가 별도로 적합한 `C=np.inf` 모델과 같은 절편·기울기를 반환하는지 다음 두 비교로 검증한다.

```python
self.assertAlmostEqual(
    metrics["calibration_intercept"],
    expected.intercept_[0],
    places=10,
)
self.assertAlmostEqual(
    metrics["calibration_slope"],
    expected.coef_[0, 0],
    places=10,
)
```

- [ ] **5단계: 분류 테스트와 전체 테스트 실행**

```bash
PYTHONPATH=src/kbo_pipeline \
python3 -m unittest tests.test_classifier -v

PYTHONPATH=src/kbo_pipeline:scripts/model:scripts/ops:scripts/collect:scripts/processing \
python3 -m unittest discover -s tests -v
```

- [ ] **6단계: Git 저장소인 경우 작업 5 커밋**

```bash
git add src/kbo_pipeline/classifier_model.py tests/test_classifier.py
git commit -m "fix: 확률 보정 로지스틱 정규화 제거"
```

---

### 작업 6: 불펜 K/BB 경계값과 수집 CSV 원자적 저장 수정

**파일:**

- 수정: `src/kbo_pipeline/feature_matrix_v9.py`
- 생성: `src/kbo_pipeline/io_utils.py`
- 수정: `scripts/collect/collect_schedule.py`
- 수정: `scripts/collect/collect_lineups.py`
- 수정: `scripts/collect/collect_rosters.py`
- 수정: `tests/test_asof_features.py`
- 생성: `tests/test_io_utils.py`

**인터페이스:**

- 새 함수: `_bounded_kbb(strikeouts: pd.Series, walks: pd.Series) -> pd.Series`
- 새 함수: `atomic_to_csv(frame: pd.DataFrame, path: Path, *, encoding: str = "utf-8-sig") -> None`
- K/BB 규칙: 볼넷이 양수면 실제 비율, 볼넷 0·삼진 양수면 상한 10.0, 둘 다 0이면 사전값 2.0, 최종 범위 0.25~10.0

- [ ] **1단계: K/BB 경계값 테스트 작성**

```python
def test_zero_walk_bullpen_kbb_distinguishes_strikeouts(self):
    from feature_matrix_v9 import _bounded_kbb

    result = _bounded_kbb(
        pd.Series([5.0, 0.0, 8.0]),
        pd.Series([0.0, 0.0, 2.0]),
    )

    self.assertEqual(result.tolist(), [10.0, 2.0, 4.0])
```

- [ ] **2단계: K/BB 테스트가 실패하는지 확인**

```bash
PYTHONPATH=src/kbo_pipeline \
python3 -m unittest tests.test_asof_features.AsofFeatureTests.test_zero_walk_bullpen_kbb_distinguishes_strikeouts -v
```

- [ ] **3단계: 벡터화된 K/BB 구현**

```python
def _bounded_kbb(
    strikeouts: pd.Series,
    walks: pd.Series,
) -> pd.Series:
    so = pd.to_numeric(strikeouts, errors="coerce").fillna(0)
    bb = pd.to_numeric(walks, errors="coerce").fillna(0)
    ratio = np.select(
        [bb.gt(0), so.gt(0)],
        [so / bb.where(bb.gt(0), 1.0), 10.0],
        default=2.0,
    )
    return pd.Series(ratio, index=so.index, dtype=float).clip(0.25, 10.0)
```

`_bullpen_features()`의 기존 나눗셈과 `fillna(2.0)`을 이 함수 호출로 교체한다.

- [ ] **4단계: 원자적 저장 실패 보존 테스트 작성**

`tests/test_io_utils.py`에서 기존 파일을 만든 뒤 `DataFrame.to_csv`를 모의 실패시키고, 기존 내용이 그대로 남으며 임시 파일도 제거되는지 검증한다.

```python
def test_atomic_to_csv_preserves_existing_file_on_write_failure(self):
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from unittest.mock import patch

    from io_utils import atomic_to_csv

    with TemporaryDirectory() as directory:
        path = Path(directory) / "data.csv"
        path.write_text("old\n", encoding="utf-8")
        frame = pd.DataFrame({"value": [1]})
        with patch.object(
            pd.DataFrame,
            "to_csv",
            side_effect=OSError("쓰기 실패"),
        ):
            with self.assertRaisesRegex(OSError, "쓰기 실패"):
                atomic_to_csv(frame, path)

        self.assertEqual(path.read_text(encoding="utf-8"), "old\n")
        self.assertEqual(list(Path(directory).glob(".data.csv.*.tmp")), [])
```

- [ ] **5단계: 원자적 CSV 쓰기 구현**

```python
"""수집 산출물을 같은 파일시스템에서 원자적으로 교체한다."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd


def atomic_to_csv(
    frame: pd.DataFrame,
    path: Path,
    *,
    encoding: str = "utf-8-sig",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
        encoding=encoding,
        newline="",
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        frame.to_csv(temporary, index=False, encoding=encoding)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
```

- [ ] **6단계: 세 수집 스크립트의 직접 덮어쓰기 교체**

`collect_schedule.py`, `collect_lineups.py`, `collect_rosters.py`에서 출력 파일을 대상으로 하는 최종 및 배치 `to_csv` 호출을 `atomic_to_csv(df_final, output_file)` 형태로 교체한다. 저장 직후 다시 읽는 기존 흐름은 유지한다.

- [ ] **7단계: 작업 6 테스트 실행**

```bash
PYTHONPATH=src/kbo_pipeline:scripts/collect \
python3 -m unittest tests.test_asof_features tests.test_io_utils -v
```

예상 결과: 볼넷 0·삼진 양수는 10.0이고, 임시 쓰기 실패 후 기존 CSV가 보존된다.

- [ ] **8단계: Git 저장소인 경우 작업 6 커밋**

```bash
git add src/kbo_pipeline/feature_matrix_v9.py \
  src/kbo_pipeline/io_utils.py \
  scripts/collect/collect_schedule.py \
  scripts/collect/collect_lineups.py \
  scripts/collect/collect_rosters.py \
  tests/test_asof_features.py \
  tests/test_io_utils.py
git commit -m "fix: 불펜 비율 경계값과 수집 파일 원자성 보장"
```

---

### 작업 7: 경기 결과의 실제 가용시각 계약 도입

**파일:**

- 수정: `scripts/collect/collect_schedule.py`
- 수정: `src/kbo_pipeline/game_time.py`
- 수정: `src/kbo_pipeline/feature_matrix_v9.py`
- 수정: `src/kbo_pipeline/fallback_recent10.py`
- 수정: `tests/test_game_datetime.py`
- 수정: `tests/test_fallback_recent10.py`
- 수정: `tests/test_asof_features.py`

**인터페이스:**

- 새 원천 열: `result_observed_at`
- 새 표준 열: `result_available_datetime`
- 일정에 미완료 상태로 한 번 이상 저장된 경기가 이후 점수 완성 상태로 바뀌면, 그 최초 수집 시각을 `result_observed_at`으로 영구 보존한다.
- 처음 발견될 때 이미 점수가 완성된 과거 경기에는 현재 시각을 역기록하지 않고 과거 데이터용 보수적 가용시각을 적용한다.
- 과거 데이터에 관측 시각이 없으면 서울 기준 경기 다음 날 00:00을 보수적 가용시각으로 사용한다.
- 재개 시각이 있으면 원래 경기일이 아니라 재개일 다음 날 00:00을 과거 데이터의 보수적 가용시각으로 사용한다.

- [ ] **1단계: 진행 중 점수가 대체 모델에서 제외되는 테스트 작성**

```python
def test_recent_probability_requires_result_availability_before_cutoff(self):
    from fallback_recent10 import recent_ten_home_probability

    history = pd.DataFrame(
        {
            "game_datetime": pd.to_datetime(
                [
                    "2026-04-01T18:30:00+09:00",
                    "2026-04-02T18:30:00+09:00",
                    "2026-04-03T18:30:00+09:00",
                    "2026-04-04T18:30:00+09:00",
                    "2026-04-05T18:30:00+09:00",
                    "2026-04-06T18:30:00+09:00",
                ],
                utc=True,
            ),
            "result_available_datetime": pd.to_datetime(
                [
                    "2026-04-01T22:30:00+09:00",
                    "2026-04-02T22:30:00+09:00",
                    "2026-04-03T22:30:00+09:00",
                    "2026-04-04T22:30:00+09:00",
                    "2026-04-05T22:30:00+09:00",
                    "2026-04-06T22:30:00+09:00",
                ],
                utc=True,
            ),
            "homeTeam": [1, 1, 1, 1, 1, 1],
            "awayTeam": [2, 2, 2, 2, 2, 2],
            "homeScore": [3.0, 3.0, 3.0, 1.0, 1.0, 1.0],
            "awayScore": [1.0, 1.0, 1.0, 3.0, 3.0, 3.0],
        }
    )

    probability = recent_ten_home_probability(
        history,
        home_team=1,
        away_team=2,
        feature_cutoff_datetime=pd.Timestamp(
            "2026-04-06T20:00:00+09:00"
        ),
        league_home_win_rate=0.54,
    )

    self.assertAlmostEqual(probability, 4 / 7)
```

- [ ] **2단계: 기존 시작시각 필터 때문에 테스트가 실패하는지 확인**

```bash
PYTHONPATH=src/kbo_pipeline \
python3 -m unittest tests.test_fallback_recent10.RecentTenFallbackTests.test_recent_probability_requires_result_availability_before_cutoff -v
```

- [ ] **3단계: 일정 수집 시 최초 결과 관측시각 보존**

`collect_schedule.py`에서 이전 수집에는 점수가 없었지만 새 응답에는 점수가 모두 존재하는 상태 전이만 탐지한다. 기존 관측 시각은 이후 수집에서도 유지하고, 처음 발견될 때 이미 완료된 과거 경기에는 현재 시각을 기록하지 않는다.

```python
fetched_at = pd.Timestamp.now(tz="Asia/Seoul").isoformat()

if not existing_df.empty:
    if "result_observed_at" not in existing_df.columns:
        existing_df["result_observed_at"] = pd.NA
    previous = existing_df[
        ["s_no", "homeScore", "awayScore", "result_observed_at"]
    ].rename(
        columns={
            "homeScore": "_previous_home_score",
            "awayScore": "_previous_away_score",
            "result_observed_at": "_previous_result_observed_at",
        }
    )
    previous["_known_before"] = True
    new_df = new_df.merge(
        previous,
        on="s_no",
        how="left",
    )
    current_scored = (
        new_df["homeScore"].notna()
        & new_df["awayScore"].notna()
    )
    previous_scored = (
        new_df["_previous_home_score"].notna()
        & new_df["_previous_away_score"].notna()
    )
    transitioned = (
        new_df["_known_before"].fillna(False)
        & current_scored
        & ~previous_scored
    )
    new_df["result_observed_at"] = new_df[
        "_previous_result_observed_at"
    ]
    new_df.loc[
        transitioned & new_df["result_observed_at"].isna(),
        "result_observed_at",
    ] = fetched_at
    new_df.drop(
        columns=[
            "_previous_home_score",
            "_previous_away_score",
            "_previous_result_observed_at",
            "_known_before",
        ],
        inplace=True,
    )
else:
    new_df["result_observed_at"] = pd.NA
```

이 저장도 작업 6의 `atomic_to_csv()`를 사용한다.

- [ ] **4단계: 결과 가용시각 표준화 구현**

`game_time.py`에 다음 함수를 추가한다.

```python
def build_result_available_datetime(games: pd.DataFrame) -> pd.Series:
    """결과가 예측 입력으로 사용 가능한 최초 시각을 반환한다."""

    game_time = _unix_to_seoul(games["gameDate"])
    observed = pd.to_datetime(
        games.get(
            "result_observed_at",
            pd.Series(pd.NaT, index=games.index),
        ),
        errors="coerce",
        utc=True,
    ).dt.tz_convert(SEOUL_TIMEZONE)
    resume = _unix_to_seoul(
        games.get(
            "gameDateResume",
            pd.Series(0, index=games.index),
        ).replace(0, np.nan)
    )
    legacy_reference = resume.fillna(game_time)
    legacy_available = (
        legacy_reference.dt.normalize()
        + pd.Timedelta(days=1)
    )
    scored = games["homeScore"].notna() & games["awayScore"].notna()
    return observed.fillna(legacy_available).where(scored)
```

`build_game_datetime_reference()`의 반환 열에 `result_available_datetime`을 포함한다.

- [ ] **5단계: 대체 모델을 가용시각 기준으로 변경**

`fallback_recent10.py`의 `_team_recent_games()`와 `backtest_recent_ten()`은 `game_datetime` 대신 `result_available_datetime < cutoff`으로 완료 결과를 고른다. 정렬도 결과 가용시각을 우선하고, 동률이면 `s_no`를 사용한다.

기존 `tests/test_fallback_recent10.py`의 모든 이력 입력에도 각 경기 이후 시각의 `result_available_datetime`을 명시하여 새로운 필수 입력 계약을 고정한다. `backtest_recent_ten()`의 필수 열 검사에도 `result_available_datetime`을 추가한다.

- [ ] **6단계: 선수 누적 이벤트도 결과 가용시각을 사용**

`feature_matrix_v9.py`의 `_prepare_events()`가 선수 일별 기록의 `event_datetime`으로 경기 시작시각이 아니라 `result_available_datetime`을 사용하도록 변경한다.

```python
reference = games[
    ["s_no", "result_available_datetime", "year"]
].rename(
    columns={
        "s_no": "s_no_key",
        "result_available_datetime": "event_datetime",
    }
)
```

이 변경으로 서스펜디드 경기와 당일 진행 중 경기의 최종 선수 기록도 실제 가용시각 전 피처에서 제외된다.

- [ ] **7단계: 과거·신규·재개 경기 경계 테스트 추가**

다음 세 경우를 `tests/test_game_datetime.py`와 `tests/test_asof_features.py`에 추가한다.

1. `result_observed_at`이 있으면 정확히 그 시각을 사용한다.
2. 과거 일반 경기는 다음 날 00:00부터 결과를 사용한다.
3. `gameDateResume`이 있으면 재개일 다음 날 00:00 전에는 결과를 사용하지 않는다.

- [ ] **8단계: 작업 7 테스트 실행**

```bash
PYTHONPATH=src/kbo_pipeline \
python3 -m unittest \
  tests.test_game_datetime \
  tests.test_fallback_recent10 \
  tests.test_asof_features -v
```

예상 결과: 점수가 채워져 있어도 `result_available_datetime` 이후에만 최근 경기와 선수 누적 이벤트로 사용된다.

- [ ] **9단계: Git 저장소인 경우 작업 7 커밋**

```bash
git add scripts/collect/collect_schedule.py \
  src/kbo_pipeline/game_time.py \
  src/kbo_pipeline/feature_matrix_v9.py \
  src/kbo_pipeline/fallback_recent10.py \
  tests/test_game_datetime.py \
  tests/test_fallback_recent10.py \
  tests/test_asof_features.py
git commit -m "fix: 경기 결과 가용시각으로 진행 중 결과 누수 차단"
```

---

## 최종 통합 검증

- [ ] **1단계: 전체 단위 테스트 실행**

```bash
PYTHONPATH=src/kbo_pipeline:scripts/model:scripts/ops:scripts/collect:scripts/processing \
python3 -m unittest discover -s tests -v
```

성공 기준: 실패와 오류가 0건이다.

- [ ] **2단계: 로컬 원천 데이터로 피처 재생성**

```bash
PYTHONPATH=src/kbo_pipeline:src \
python3 -m scripts.build.create_feature_matrix_v9
```

성공 기준:

- `data/final/final_training_set_v9.csv`가 생성된다.
- 동일 경기의 미래 기록을 변경하는 회귀 테스트가 계속 통과한다.
- 필수 피처 결측과 중복 경기 ID 검증이 통과한다.

- [ ] **3단계: 모델 재학습과 비교 산출물 재생성**

```bash
PYTHONPATH=src/kbo_pipeline:src \
python3 -m scripts.model.tune_hyperparameters

PYTHONPATH=src/kbo_pipeline:src \
python3 -m scripts.model.train_classifier

PYTHONPATH=src/kbo_pipeline:src \
python3 -m scripts.model.train_score_models

PYTHONPATH=src/kbo_pipeline:src \
python3 -m scripts.model.compare_models
```

성공 기준:

- 개발 폴드 지표는 모두 시간 분리된 보정 확률로 계산된다.
- `best_model_metadata.json`은 2026 지표를 선택에 사용하지 않았다고 기록한다.
- `best_model.joblib`은 모델 계열과 관계없이 `feature_columns`와 `predict_proba()`를 제공한다.

- [ ] **4단계: 고정 운영 모델 백테스트 재실행**

```bash
PYTHONPATH=src/kbo_pipeline:src \
python3 -m scripts.model.backtest
```

성공 기준:

- 분류 모델과 득점 모델 래퍼 중 어느 것이 선택되어도 같은 명령으로 완료된다.
- 2026년 결과는 모델 선택이 끝난 뒤 한 번만 최종 평가에 사용된다.
- 이전 산출물과 새 산출물의 로그 손실·브라이어 점수·보정 지표는 마크다운 표로 비교한다.

- [ ] **5단계: 운영 전 수동 점검**

실제 API 전송 없이 실시간 스크립트의 모델 로드와 당일 피처 생성까지만 별도 시험 환경에서 확인한다. 외부 전송은 사용자 승인과 유효한 API 자격증명이 있을 때만 수행한다.

---

## 완료 정의

다음 조건을 모두 만족해야 이 계획을 완료로 판정한다.

- 시즌 전체 상수를 바꿔도 과거 경기 피처가 변하지 않는다.
- 최종 테스트 지표를 바꿔도 선택되는 운영 모델이 변하지 않는다.
- 저장된 모든 운영 모델이 동일한 `predict_proba()` 계약을 만족한다.
- 개발 폴드 보정기는 검증 정답을 학습하지 않는다.
- 보정 절편과 기울기는 무정규화 로지스틱 회귀로 계산된다.
- 볼넷 0·삼진 양수인 불펜의 K/BB는 10.0으로 처리된다.
- 수집 저장 실패 시 기존 CSV가 손상되지 않는다.
- 경기 결과와 선수 기록은 결과 가용시각 이후에만 피처에 포함된다.
- 전체 단위 테스트와 오프라인 재학습·백테스트가 통과한다.
