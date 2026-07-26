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
    """임시 파일에 쓴 뒤 원자적으로 교체하여 기존 파일 손상을 방지한다."""
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
