"""Client for the NVD CVE API 2.0.

Separate from the document adapter on purpose: this module knows about HTTP,
rate limits and caching, and nothing about :class:`~threatrag.domain.models.
Document`. That split is what makes the terms-of-use obligations testable --
they are properties of *how we call the API*, and they can be asserted against
a fake transport without ever touching the network.

The NVD terms of use bind this code in four concrete ways, and each is
implemented here rather than left to the caller's discipline:

* **Rate limits.** Requests are serialised through one limiter with no
  concurrency, defaulting to the interval NVD's own developer guidance
  recommends rather than the fastest the quota technically allows. There is no
  IP rotation, proxying or any other means of exceeding the published limits --
  the TOU treats attempting to circumvent them as grounds for a block.
* **Backoff, then surrender.** A 429 or 503 is retried a bounded number of
  times with exponential backoff, honouring ``Retry-After`` when the server
  sends one. After that the fetch fails loudly instead of hammering.
* **The key is never exposed.** It travels in a header, never a query string,
  so it cannot leak through a URL echoed into a log, an exception or a cached
  filename. :func:`NvdClient.describe` exists so callers can report whether a
  key is in use without being able to print it.
* **Attribution.** :data:`NVD_ATTRIBUTION` is defined here, once, so the UI,
  the README and the M3 report all quote the required notice identically.

Caching is not just politeness. Every response is written to disk under a
deterministic name, so raising the CVE cap later re-fetches only the delta, an
interrupted run resumes, and re-running the ingest to compare chunkers costs
zero requests.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# Required by the NVD terms of use, displayed prominently by anything built on
# this client. Deliberately not embedded in document text: anything in a
# document body is chunked, embedded and retrievable, and a legal notice
# surfacing as though it were threat intelligence is both noise and a small
# injection surface.
NVD_ATTRIBUTION = "This product uses the NVD API but is not endorsed or certified by the NVD."

# NVD rejects a publication window wider than 120 days.
MAX_WINDOW = timedelta(days=120)

# The API caps a page at 2000 results.
MAX_PAGE_SIZE = 2000

RETRY_STATUS = frozenset({403, 429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class RateLimit:
    """A published NVD quota, expressed as the interval it implies.

    NVD documents its limits as "N requests per rolling 30 seconds" but advises
    sleeping between consecutive requests rather than spending the whole budget
    at once, because a burst that fits the quota can still trip the rolling
    window. Pacing at ``window / requests`` is that advice, and it is what makes
    a long unattended fetch survive to the end.
    """

    requests: int
    window_seconds: float = 30.0

    @property
    def interval(self) -> float:
        return self.window_seconds / self.requests


# Published at nvd.nist.gov/developers: 5 requests per 30s without a key, 50
# with one. A key is optional throughout -- it makes the fetch faster and
# changes nothing else.
PUBLIC_RATE = RateLimit(requests=5)
KEYED_RATE = RateLimit(requests=50)


class NvdError(RuntimeError):
    """A request failed after the client exhausted its bounded retries."""


class _Pacer:
    """Serialises requests to at most one per ``interval`` seconds."""

    def __init__(self, interval: float, sleep: Any = time.sleep) -> None:
        self._interval = interval
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = time.monotonic()
        if self._last is not None:
            remaining = self._interval - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = time.monotonic()


def _iso(moment: datetime) -> str:
    """NVD expects ``YYYY-MM-DDTHH:MM:SS.mmm`` with no timezone suffix."""
    return moment.strftime("%Y-%m-%dT%H:%M:%S.000")


class NvdClient:
    """Rate-limited, caching, retrying access to the NVD CVE API."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        max_attempts: int = 4,
        sleep: Any = time.sleep,
        rate: RateLimit | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        # Normalise "" to None so an untouched .env placeholder is treated as
        # absent rather than as a key that would be rejected on every request.
        self._api_key = (api_key or "").strip() or None
        self._rate = rate or (KEYED_RATE if self._api_key else PUBLIC_RATE)
        self._pacer = _Pacer(self._rate.interval, sleep=sleep)
        self._sleep = sleep
        self._max_attempts = max_attempts
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=httpx.Timeout(30.0, read=60.0))
        self.requests_made = 0

    # -- reporting -------------------------------------------------------

    def describe(self) -> str:
        """A one-line summary safe to print: says *whether* a key is set, never what it is."""
        source = "registered key" if self._api_key else "public (no key)"
        return (
            f"NVD rate limit: {source}, {self._rate.requests} requests / "
            f"{self._rate.window_seconds:.0f}s"
        )

    @property
    def has_key(self) -> bool:
        return self._api_key is not None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> NvdClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- transport -------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {
            # Identifies the project to NVD's operators, as their guidance asks.
            "User-Agent": "threatrag/0.1 (research; secure-rag-threat-intelligence)",
            "Accept": "application/json",
        }
        if self._api_key:
            # Header, never a query parameter: a URL ends up in logs and
            # exception messages, and the TOU forbids sharing the key.
            headers["apiKey"] = self._api_key
        return headers

    def _request(self, params: dict[str, str | int]) -> dict[str, Any]:
        last_status: int | None = None
        for attempt in range(1, self._max_attempts + 1):
            self._pacer.wait()
            self.requests_made += 1
            response = self._client.get(NVD_API_URL, params=params, headers=self._headers())

            if response.status_code == 200:
                payload: dict[str, Any] = response.json()
                return payload

            last_status = response.status_code
            if response.status_code not in RETRY_STATUS or attempt == self._max_attempts:
                break

            # Honour an explicit Retry-After; otherwise back off exponentially
            # from the pacing interval. Bounded either way -- retrying a rate
            # limit indefinitely is precisely the "attempt to exceed or
            # circumvent" the terms of use warn about.
            retry_after = response.headers.get("Retry-After")
            delay = (
                float(retry_after)
                if retry_after and retry_after.isdigit()
                else self._rate.interval * (2**attempt)
            )
            self._sleep(delay)

        raise NvdError(
            f"NVD request failed with HTTP {last_status} after {self._max_attempts} attempts. "
            "Waiting and re-running will resume from the cache."
        )

    # -- caching ---------------------------------------------------------

    def _cached(
        self, name: str, params: dict[str, str | int], *, force: bool = False
    ) -> dict[str, Any]:
        path = self._cache_dir / f"{name}.json"
        if path.exists() and not force:
            with path.open(encoding="utf-8") as handle:
                cached: dict[str, Any] = json.load(handle)
            return cached

        payload = self._request(params)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: an interrupted fetch must not leave a truncated
        # file that later looks like a valid cache hit.
        tmp = path.with_suffix(".json.part")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        tmp.replace(path)
        return payload

    # -- queries ---------------------------------------------------------

    def get_cve(self, cve_id: str, *, force: bool = False) -> dict[str, Any] | None:
        """One CVE record, or ``None`` when NVD has no such entry.

        ATT&CK cites CVE ids that NVD has since rejected or never held, so a
        miss is normal data, not an error.
        """
        payload = self._cached(f"cve/{cve_id.upper()}", {"cveId": cve_id.upper()}, force=force)
        vulnerabilities = payload.get("vulnerabilities") or []
        if not vulnerabilities:
            return None
        record: dict[str, Any] = vulnerabilities[0].get("cve", {})
        return record or None

    def iter_window(
        self,
        start: datetime,
        end: datetime,
        *,
        severity: str | None = None,
        limit: int | None = None,
        force: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """CVEs published in ``[start, end)``, paged, newest window first.

        ``limit`` stops the walk early; the caller uses it to honour a corpus
        cap without downloading pages it will discard.
        """
        if end - start > MAX_WINDOW:
            raise ValueError(f"NVD allows at most {MAX_WINDOW.days} days per window")

        yielded = 0
        start_index = 0
        while True:
            params: dict[str, str | int] = {
                "pubStartDate": _iso(start),
                "pubEndDate": _iso(end),
                "resultsPerPage": MAX_PAGE_SIZE,
                "startIndex": start_index,
            }
            if severity:
                params["cvssV3Severity"] = severity.upper()

            name = f"window/{start:%Y%m%d}-{end:%Y%m%d}-{severity or 'all'}-{start_index}"
            payload = self._cached(name, params, force=force)

            vulnerabilities = payload.get("vulnerabilities") or []
            for entry in vulnerabilities:
                record = entry.get("cve")
                if not record:
                    continue
                yield record
                yielded += 1
                if limit is not None and yielded >= limit:
                    return

            start_index += len(vulnerabilities)
            total = int(payload.get("totalResults", 0))
            if not vulnerabilities or start_index >= total:
                return


def windows_back_from(end: datetime, months: int) -> Iterator[tuple[datetime, datetime]]:
    """Walk backwards from ``end`` in chunks NVD will accept, newest first.

    The caller passes a fixed ``end`` from config rather than ``now()``. That is
    what keeps the corpus reproducible: a window anchored to the current date
    would quietly change the document set between runs, and every retrieval
    number measured against it would silently stop being comparable.
    """
    span = timedelta(days=int(months * 30.44))
    earliest = end - span
    cursor = end
    while cursor > earliest:
        window_start = max(earliest, cursor - MAX_WINDOW)
        yield window_start, cursor
        cursor = window_start
