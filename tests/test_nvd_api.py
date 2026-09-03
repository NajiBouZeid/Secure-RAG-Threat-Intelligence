"""Tests for the NVD client.

Several of these assert terms-of-use obligations rather than functionality: that
the key never reaches a URL, that retries are bounded, that pacing follows the
published limit. Those are the properties most likely to be broken by a
well-meaning later edit ("make the fetch faster"), and the ones with a
consequence worse than a failing test if they are.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from threatrag.ingest.sources.nvd_api import (
    KEYED_RATE,
    PUBLIC_RATE,
    NvdClient,
    NvdError,
    RateLimit,
    windows_back_from,
)


def _cve(cve_id: str) -> dict[str, Any]:
    return {"id": cve_id, "descriptions": [{"lang": "en", "value": f"About {cve_id}"}]}


def _page(records: list[dict[str, Any]], total: int, start_index: int = 0) -> dict[str, Any]:
    return {
        "resultsPerPage": len(records),
        "startIndex": start_index,
        "totalResults": total,
        "vulnerabilities": [{"cve": record} for record in records],
    }


class Recorder:
    """Collects every request so the tests can assert on how we called NVD."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self._responses = responses
        self.slept: list[float] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self._responses) - 1)
        return self._responses[index]

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)


def _build(tmp_path: Path, recorder: Recorder, **kwargs: Any) -> NvdClient:
    return NvdClient(tmp_path / "nvd", client=recorder.client(), sleep=recorder.sleep, **kwargs)


def test_key_travels_in_a_header_never_the_url(tmp_path: Path) -> None:
    """A URL leaks into logs and exception text; the terms of use forbid sharing the key."""
    recorder = Recorder([httpx.Response(200, json=_page([_cve("CVE-2024-3094")], 1))])
    client = _build(tmp_path, recorder, api_key="SECRET-KEY-VALUE")

    client.get_cve("CVE-2024-3094")

    request = recorder.requests[0]
    assert request.headers["apiKey"] == "SECRET-KEY-VALUE"
    assert "SECRET-KEY-VALUE" not in str(request.url)
    assert "apikey" not in str(request.url).lower()


def test_describe_reports_the_limit_without_revealing_the_key(tmp_path: Path) -> None:
    recorder = Recorder([httpx.Response(200, json=_page([], 0))])
    client = _build(tmp_path, recorder, api_key="SECRET-KEY-VALUE")

    summary = client.describe()

    assert "SECRET-KEY-VALUE" not in summary
    assert "registered key" in summary


def test_blank_key_is_treated_as_absent(tmp_path: Path) -> None:
    """An untouched .env placeholder must not be sent as a key NVD would reject."""
    recorder = Recorder([httpx.Response(200, json=_page([], 0))])
    client = _build(tmp_path, recorder, api_key="   ")

    assert client.has_key is False
    client.get_cve("CVE-2024-3094")
    assert "apiKey" not in recorder.requests[0].headers


def test_rate_defaults_follow_the_published_limits(tmp_path: Path) -> None:
    recorder = Recorder([httpx.Response(200, json=_page([], 0))])

    assert _build(tmp_path, recorder)._rate == PUBLIC_RATE
    assert _build(tmp_path, recorder, api_key="k")._rate == KEYED_RATE
    # NVD's guidance: pace requests rather than spending the quota in a burst.
    assert PUBLIC_RATE.interval == pytest.approx(6.0)
    assert KEYED_RATE.interval == pytest.approx(0.6)


def test_consecutive_requests_are_paced(tmp_path: Path) -> None:
    recorder = Recorder([httpx.Response(200, json=_page([_cve("CVE-1")], 1))])
    client = _build(tmp_path, recorder, rate=RateLimit(requests=5))

    client.get_cve("CVE-1")
    client.get_cve("CVE-2")

    # The first request does not wait; the second is held back to the interval.
    assert len(recorder.slept) == 1
    assert recorder.slept[0] == pytest.approx(6.0, abs=0.5)


def test_retries_are_bounded_then_the_fetch_fails_loudly(tmp_path: Path) -> None:
    """Retrying a rate limit forever is what the terms of use call circumvention."""
    recorder = Recorder([httpx.Response(429)])
    client = _build(tmp_path, recorder, max_attempts=3)

    with pytest.raises(NvdError, match="after 3 attempts"):
        client.get_cve("CVE-2024-3094")

    assert len(recorder.requests) == 3


def test_retry_after_header_is_honoured(tmp_path: Path) -> None:
    recorder = Recorder(
        [
            httpx.Response(429, headers={"Retry-After": "17"}),
            httpx.Response(200, json=_page([_cve("CVE-1")], 1)),
        ]
    )
    client = _build(tmp_path, recorder)

    client.get_cve("CVE-1")

    assert 17.0 in recorder.slept


def test_a_client_error_is_not_retried(tmp_path: Path) -> None:
    recorder = Recorder([httpx.Response(404)])
    client = _build(tmp_path, recorder)

    with pytest.raises(NvdError):
        client.get_cve("CVE-1")

    assert len(recorder.requests) == 1


def test_responses_are_cached_so_a_rerun_costs_no_requests(tmp_path: Path) -> None:
    recorder = Recorder([httpx.Response(200, json=_page([_cve("CVE-2024-3094")], 1))])
    client = _build(tmp_path, recorder)

    first = client.get_cve("CVE-2024-3094")
    second = client.get_cve("CVE-2024-3094")

    assert first == second
    assert client.requests_made == 1
    assert (tmp_path / "nvd" / "cve" / "CVE-2024-3094.json").exists()


def test_cache_survives_a_new_client(tmp_path: Path) -> None:
    """Raising the corpus cap later must re-fetch only the delta."""
    recorder = Recorder([httpx.Response(200, json=_page([_cve("CVE-1")], 1))])
    _build(tmp_path, recorder).get_cve("CVE-1")

    fresh = Recorder([httpx.Response(500)])
    client = _build(tmp_path, fresh)
    assert client.get_cve("CVE-1") is not None
    assert fresh.requests == []


def test_an_unknown_cve_is_data_not_an_error(tmp_path: Path) -> None:
    recorder = Recorder([httpx.Response(200, json=_page([], 0))])
    client = _build(tmp_path, recorder)

    assert client.get_cve("CVE-1999-0000") is None


def test_window_pages_until_the_total_is_reached(tmp_path: Path) -> None:
    first = _page([_cve(f"CVE-A-{i}") for i in range(3)], total=5)
    second = _page([_cve(f"CVE-B-{i}") for i in range(2)], total=5, start_index=3)
    recorder = Recorder([httpx.Response(200, json=first), httpx.Response(200, json=second)])
    client = _build(tmp_path, recorder)

    found = list(client.iter_window(datetime(2026, 1, 1), datetime(2026, 3, 1)))

    assert len(found) == 5
    assert len(recorder.requests) == 2


def test_window_limit_stops_early_instead_of_downloading_discarded_pages(tmp_path: Path) -> None:
    page = _page([_cve(f"CVE-A-{i}") for i in range(10)], total=1000)
    recorder = Recorder([httpx.Response(200, json=page)])
    client = _build(tmp_path, recorder)

    found = list(client.iter_window(datetime(2026, 1, 1), datetime(2026, 3, 1), limit=4))

    assert len(found) == 4
    assert len(recorder.requests) == 1


def test_window_wider_than_nvd_allows_is_rejected(tmp_path: Path) -> None:
    recorder = Recorder([httpx.Response(200, json=_page([], 0))])
    client = _build(tmp_path, recorder)

    with pytest.raises(ValueError, match="120 days"):
        list(client.iter_window(datetime(2025, 1, 1), datetime(2026, 1, 1)))


def test_severity_filter_is_sent_upstream(tmp_path: Path) -> None:
    recorder = Recorder([httpx.Response(200, json=_page([], 0))])
    client = _build(tmp_path, recorder)

    list(client.iter_window(datetime(2026, 1, 1), datetime(2026, 2, 1), severity="high"))

    assert "cvssV3Severity=HIGH" in str(recorder.requests[0].url)


def test_windows_walk_backwards_within_the_api_limit() -> None:
    end = datetime(2026, 9, 1)
    windows = list(windows_back_from(end, months=18))

    assert windows[0][1] == end
    assert all((stop - start).days <= 120 for start, stop in windows)
    # Contiguous and strictly backwards: no gap can silently drop CVEs.
    for earlier, later in zip(windows[1:], windows[:-1], strict=True):
        assert earlier[1] == later[0]
    assert (end - windows[-1][0]).days >= 540
