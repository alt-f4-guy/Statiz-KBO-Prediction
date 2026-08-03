"""인증 서명, 제한 시간과 제한된 재시도를 일관되게 적용하는 API 클라이언트."""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from typing import Any, Callable, Mapping
from urllib.parse import quote, urlencode

import requests


LOGGER = logging.getLogger(__name__)


def extract_players(payload: Any) -> list[dict[str, Any]]:
    """중첩된 라인업 응답에서 선수 행만 순서대로 추출한다."""

    if isinstance(payload, dict):
        if "p_no" in payload:
            return [payload]
        return [
            player
            for key, value in payload.items()
            if key not in {"result_cd", "result_msg", "update_time"}
            for player in extract_players(value)
        ]
    if isinstance(payload, list):
        return [player for item in payload for player in extract_players(item)]
    return []


class StatizAPIError(RuntimeError):
    """Statiz API 요청 또는 응답 계약이 실패한 경우 발생한다."""


class StatizAPI:
    def __init__(
        self,
        api_key: str,
        secret: str,
        *,
        timeout: tuple[float, float] = (3.05, 30.0),
        max_retries: int = 3,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self.secret = secret
        self.base_url = "https://api.statiz.co.kr/baseballApi"
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.sleep = sleep

    def _make_signature(
        self,
        method: str,
        path: str,
        query_string: str,
        timestamp: str,
    ) -> str:
        payload = f"{method}|{path}|{query_string}|{timestamp}"
        return hmac.new(
            self.secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _headers(
        self,
        method: str,
        path: str,
        query_string: str,
        *,
        content_type: str | None = None,
    ) -> dict[str, str]:
        timestamp = str(int(time.time()))
        headers = {
            "X-API-KEY": self.api_key,
            "X-TIMESTAMP": timestamp,
            "X-SIGNATURE": self._make_signature(
                method, path, query_string, timestamp
            ),
            "Accept": "application/json",
        }
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    @staticmethod
    def _query_string(params: Mapping[str, Any]) -> tuple[dict[str, str], str]:
        safe = "-_.!~*'()"
        normalized = {key: str(params[key]) for key in sorted(params)}
        return normalized, urlencode(normalized, quote_via=quote, safe=safe)

    def _decode(self, response: requests.Response, path: str) -> Any:
        if response.status_code != 200:
            message = response.text[:300].replace("\n", " ")
            raise StatizAPIError(
                f"{path} 요청 실패: HTTP {response.status_code} {message}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise StatizAPIError(f"{path} 응답이 JSON 형식이 아닙니다.") from exc

    @staticmethod
    def _rate_limit_delay(
        response: requests.Response,
        fallback: float,
    ) -> float:
        """429 응답의 서버 쿨다운을 읽고 유효하지 않으면 대체값을 쓴다."""

        try:
            payload = response.json()
            cooldown = float(payload["rate_limit"]["cooldown_sec"])
        except (KeyError, TypeError, ValueError):
            return float(fallback)
        if not math.isfinite(cooldown) or cooldown <= 0:
            return float(fallback)
        return min(300.0, cooldown)

    def get(self, path: str, params: Mapping[str, Any]) -> Any:
        normalized, query_string = self._query_string(params)
        url = f"{self.base_url}/{path}"

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    url,
                    params=normalized,
                    headers=self._headers("GET", path, query_string),
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise StatizAPIError(f"{path} 네트워크 요청 실패") from exc
                LOGGER.warning("%s 요청 재시도: %s/%s", path, attempt + 1, self.max_retries)
                self.sleep(2**attempt)
                continue

            if response.status_code == 429 and attempt < self.max_retries:
                delay = self._rate_limit_delay(
                    response,
                    fallback=min(60.0, 2**attempt),
                )
                LOGGER.warning(
                    "%s 요청 제한 응답, %.1f초 후 재시도합니다.",
                    path,
                    delay,
                )
                self.sleep(delay)
                continue
            return self._decode(response, path)

        raise StatizAPIError(f"{path} 요청 재시도 횟수를 초과했습니다.")

    def post(self, path: str, data: Mapping[str, Any]) -> Any:
        url = f"{self.base_url}/{path}"
        headers = self._headers(
            "POST",
            path,
            "",
            content_type="application/x-www-form-urlencoded",
        )
        try:
            response = self.session.post(
                url,
                data=dict(data),
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise StatizAPIError(f"{path} 네트워크 요청 실패") from exc
        return self._decode(response, path)
