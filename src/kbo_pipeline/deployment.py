"""운영 배포 버전과 전향적 평가 역할을 재현 가능하게 고정한다."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


ARTIFACT_FIELDS = {
    "model_path": "model_checksum",
    "data_path": "data_checksum",
    "split_manifest": "split_manifest_checksum",
}


def sha256_file(path: Path) -> str:
    """파일 내용을 스트리밍하여 SHA-256 체크섬을 반환한다."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_file(project_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError("배포 산출물 경로는 프로젝트 상대 경로여야 합니다.")
    root = project_root.resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("배포 산출물 경로가 프로젝트 밖을 가리킵니다.")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def build_deployment_context(
    project_root: Path,
    metadata: Mapping[str, Any],
    git_commit: str,
) -> dict[str, Any]:
    """Git 커밋과 핵심 산출물 체크섬으로 불변 배포 문맥을 만든다."""

    prospective_start_date = str(metadata["prospective_start_date"])
    pd.Timestamp(prospective_start_date)
    baseline_probability = float(metadata["baseline_home_probability"])
    if not 0 <= baseline_probability <= 1:
        raise ValueError("고정 홈 승률 기준선은 0과 1 사이여야 합니다.")

    context: dict[str, Any] = {
        "git_commit": str(git_commit),
        "prospective_start_date": prospective_start_date,
        "baseline_home_probability": baseline_probability,
    }
    for path_field, checksum_field in ARTIFACT_FIELDS.items():
        context[checksum_field] = sha256_file(
            _project_file(project_root, str(metadata[path_field]))
        )

    payload = json.dumps(context, sort_keys=True, separators=(",", ":"))
    context["deployment_id"] = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return context


def evaluation_role_for(
    recorded_at: pd.Timestamp,
    prospective_start_date: str,
) -> str:
    """서울 날짜 기준으로 전향적 평가 시작 전후 역할을 구분한다."""

    timestamp = pd.Timestamp(recorded_at)
    if timestamp.tzinfo is None:
        raise ValueError("recorded_at에는 시간대 정보가 필요합니다.")
    seoul_date = timestamp.tz_convert("Asia/Seoul").date()
    start_date = pd.Timestamp(prospective_start_date).date()
    return (
        "prospective_holdout"
        if seoul_date >= start_date
        else "engineering_regression"
    )
