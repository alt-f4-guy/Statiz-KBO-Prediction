# 삭제 후보 목록

아래 항목은 이번 정리에서 삭제하지 않고 보존했다. 재현성·감사 필요 여부를 확인한 뒤에만 별도로 삭제한다.

| 대상 | 현재 위치 | 삭제 전 확인할 사항 |
|---|---|---|
| 이전 옵션 구현 | `archive/legacy/Option_A/`, `archive/legacy/Option_B/` | v9 이전 결과와 구현을 다시 대조할 필요가 없는지 |
| v7 피처 생성기 | `archive/legacy/create_feature_matrix_v7.py` | v7 재현이나 피처 차이 분석이 더는 필요 없는지 |
| CatBoost 학습 로그 | `artifacts/training_logs/catboost_info/` | 현재 운영 모델의 학습 감사가 더는 필요 없는지 |
| 파이썬 캐시 | `__pycache__/` | 실행 속도 캐시이며 언제든 재생성 가능함 |

다음 항목은 삭제 후보로 분류하지 않는다.

- `data/`: 누적 원천·정형·최종 데이터셋으로 재현성과 시점 검증에 필요하다.
- `artifacts/models/`, `artifacts/tuning/`, `artifacts/evaluations/`: 현재 모델 선택과 성능 비교의 근거다.
- `.ua/`: 소유·용도가 불명확한 도구 상태 디렉터리이므로 소유자 확인 전에는 건드리지 않는다.

이번 정리에서 실제 삭제한 항목은 사용자 요청에 따른 `agent/`와 루트 `.DS_Store`뿐이다.
