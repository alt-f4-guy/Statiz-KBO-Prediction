"""파이프라인 전역 경로와 환경변수 기반 인증 설정."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FINAL_DATA_DIR = DATA_DIR / "final"
REFERENCE_DATA_DIR = DATA_DIR / "reference"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
EVALUATIONS_DIR = ARTIFACTS_DIR / "evaluations"
TUNING_DIR = ARTIFACTS_DIR / "tuning"
OPERATIONS_DIR = ARTIFACTS_DIR / "operations"


class ConfigurationError(RuntimeError):
    """필수 실행 설정이 누락된 경우 발생한다."""


@dataclass(frozen=True)
class APICredentials:
    """로그에 출력하지 않는 Statiz API 인증 정보."""

    api_key: str
    secret: str
    ptt_idx: str | None = None


def load_api_credentials(*, require_ptt_idx: bool = False) -> APICredentials:
    """환경변수에서 인증값을 읽고 네트워크 호출 전에 완전성을 검사한다."""

    api_key = os.getenv("STATIZ_API_KEY", "").strip()
    secret = os.getenv("STATIZ_SECRET", "").strip()
    ptt_idx = os.getenv("STATIZ_PTT_IDX", "").strip() or None

    missing = []
    if not api_key:
        missing.append("STATIZ_API_KEY")
    if not secret:
        missing.append("STATIZ_SECRET")
    if require_ptt_idx and not ptt_idx:
        missing.append("STATIZ_PTT_IDX")
    if missing:
        names = ", ".join(missing)
        raise ConfigurationError(f"필수 환경변수가 없습니다: {names}")

    return APICredentials(api_key=api_key, secret=secret, ptt_idx=ptt_idx)
