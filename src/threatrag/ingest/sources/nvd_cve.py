"""NVD CVE source: API records normalised into :class:`Document` objects.

Selection is deliberate rather than exhaustive. All ~250k CVEs would take hours
to embed and would swamp ATT&CK in the index without answering a single extra
question well, so the corpus is built in two passes:

* **Every CVE ATT&CK discusses.** Guaranteed relevant, and already sharing
  vocabulary with the techniques and groups already indexed.
* **A bounded recent high-severity slice**, walked backwards from a date fixed
  in config. Newest first, capped, and reproducible.

Both passes run through the same cache, so raising the cap later re-fetches
only the delta.

**On the NVD terms of use.** The terms say that content modified after being
retrieved may not then be attributed to the NVD, and this module reshapes JSON
into sectioned markdown. The line taken here is that reformatting for retrieval
is not misrepresentation, and the code is written so that claim holds up: the
description is copied verbatim and never truncated, summarised or rewritten;
only enumerated lists are bounded, and every truncation says so in the text
rather than silently cutting; and every document carries its canonical NVD URL
so the unmodified record is one click away. The required attribution notice is
displayed by the application, not embedded in document text -- anything in a
document body is chunked, embedded and retrievable, and a legal notice
surfacing as though it were threat intelligence would be both noise and a small
injection surface.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from typing import Any

from threatrag.domain.models import TLP, Document, SourceType, TrustTier
from threatrag.ingest.sources.base import section
from threatrag.ingest.sources.nvd_api import NvdClient, windows_back_from

NVD_DETAIL_URL = "https://nvd.nist.gov/vuln/detail/"

# Bounds on the enumerated sections. A widely-deployed library resolves to
# thousands of CPE entries, which would be most of the chunk budget spent on
# near-identical version strings -- text that embeds poorly and answers nothing.
# The description, which is the part that carries meaning, is never bounded.
MAX_PRODUCTS = 25
MAX_REFERENCES = 15

# NVD emits these placeholders where no CWE was assigned. They are not
# weaknesses and carry no signal.
_CWE_PLACEHOLDERS = ("NVD-CWE-noinfo", "NVD-CWE-Other")


def _english(entries: Sequence[dict[str, Any]], key: str = "value") -> str:
    for entry in entries or []:
        if entry.get("lang") == "en":
            return str(entry.get(key, "")).strip()
    return ""


def _primary_metric(record: Mapping[str, Any]) -> dict[str, Any]:
    """The best available CVSS metric, newest scoring version first.

    NVD carries v4.0, v3.1, v3.0 and v2.0 side by side and not every CVE has
    every one, so a fixed key would silently drop the score for whole slices of
    the corpus. Within a version, a Primary metric (NVD's own analysis)
    outranks a Secondary one supplied by the reporting vendor.
    """
    for key in ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        metrics = record.get("metrics", {}).get(key) or []
        if not metrics:
            continue
        ranked = sorted(metrics, key=lambda m: m.get("type") != "Primary")
        return dict(ranked[0])
    return {}


def _cwes(record: Mapping[str, Any]) -> list[str]:
    found: list[str] = []
    for weakness in record.get("weaknesses", []) or []:
        value = _english(weakness.get("description", []))
        if value and not value.startswith(_CWE_PLACEHOLDERS) and value not in found:
            found.append(value)
    return found


def _products(record: Mapping[str, Any]) -> list[str]:
    """Distinct ``vendor product`` pairs from the CPE match criteria.

    Version strings are dropped on purpose. They multiply one product into
    hundreds of near-identical lines that crowd out the description without
    making the document any more answerable.
    """
    seen: list[str] = []
    for configuration in record.get("configurations", []) or []:
        for node in configuration.get("nodes", []) or []:
            for match in node.get("cpeMatch", []) or []:
                parts = str(match.get("criteria", "")).split(":")
                if len(parts) < 6:
                    continue
                vendor, product = parts[3], parts[4]
                label = f"{vendor} {product}".replace("_", " ")
                if label not in seen:
                    seen.append(label)
    return seen


def _references(record: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for reference in record.get("references", []) or []:
        url = str(reference.get("url", "")).strip()
        if not url:
            continue
        tags = ", ".join(str(tag) for tag in reference.get("tags", []) or [])
        lines.append(f"- {url} ({tags})" if tags else f"- {url}")
    return lines


def _bounded(items: Sequence[str], limit: int, noun: str) -> str:
    """Render at most ``limit`` items, saying so when the rest are dropped.

    Marking the truncation matters beyond tidiness: an answer built on a
    silently-cut list looks complete and is not, and the terms of use are
    easier to stand behind when every departure from the record is visible.
    """
    if not items:
        return ""
    shown = list(items[:limit])
    body = "\n".join(shown)
    remaining = len(items) - len(shown)
    if remaining > 0:
        body += f"\n... and {remaining} more {noun} not shown"
    return body


def is_rejected(record: Mapping[str, Any]) -> bool:
    """Rejected and withdrawn entries carry no intelligence, only a tombstone."""
    if str(record.get("vulnStatus", "")).lower() in {"rejected", "rejected by source"}:
        return True
    return _english(record.get("descriptions", [])).lstrip().startswith("** REJECT")


def to_document(record: Mapping[str, Any], attack_ids: Sequence[str] = ()) -> Document:
    """Shape one NVD record into a retrievable document."""
    cve_id = str(record.get("id", "")).upper()
    description = _english(record.get("descriptions", []))

    metric = _primary_metric(record)
    cvss = metric.get("cvssData", {})
    score = cvss.get("baseScore")
    severity = str(cvss.get("baseSeverity", "") or "").upper()
    vector = str(cvss.get("vectorString", "") or "")
    version = str(cvss.get("version", "") or "")

    severity_lines: list[str] = []
    if score is not None:
        severity_lines.append(f"CVSS v{version} base score {score} ({severity or 'unrated'})")
    if vector:
        severity_lines.append(f"Vector: {vector}")

    cwes = _cwes(record)
    products = _products(record)
    references = _references(record)

    body = "".join(
        (
            # Verbatim and unbounded: this is the content, and the terms of use
            # are only defensible if it is reproduced rather than rewritten.
            section("Description", description),
            section("Severity", "\n".join(severity_lines)),
            section("Weaknesses", "\n".join(cwes)),
            section("Affected Products", _bounded(products, MAX_PRODUCTS, "affected products")),
            section("References", _bounded(references, MAX_REFERENCES, "references")),
            # Last: a navigational cross-reference, not an assertion that the
            # technique exploits this CVE. See AttackCtiSource.cve_mentions.
            section("Discussed by ATT&CK", ", ".join(attack_ids)),
        )
    )

    metadata: dict[str, str | list[str]] = {
        "published": str(record.get("published", "")),
        "last_modified": str(record.get("lastModified", "")),
        "vuln_status": str(record.get("vulnStatus", "")),
        "cvss_score": "" if score is None else str(score),
        "cvss_severity": severity,
        "cvss_vector": vector,
        "cwes": cwes,
        "attack_ids": list(attack_ids),
        "products_truncated": str(len(products) > MAX_PRODUCTS),
    }

    return Document(
        id=f"nvd:{cve_id}",
        title=f"{cve_id} ({severity})" if severity else cve_id,
        text=f"# {cve_id}\n\n{body}".rstrip() + "\n",
        source_type=SourceType.NVD_CVE,
        source_ref=cve_id,
        url=f"{NVD_DETAIL_URL}{cve_id}",
        # Public and structured upstream: readable by everyone, and trusted
        # enough that its content may legitimately shape an answer.
        tlp=TLP.CLEAR,
        trust_tier=TrustTier.AUTHORITATIVE,
        metadata=metadata,
    )


class NvdCveSource:
    """Selects, fetches and normalises the CVE corpus."""

    name = "nvd_cve"
    source_type = SourceType.NVD_CVE

    def __init__(
        self,
        client: NvdClient,
        *,
        attack_links: Mapping[str, Sequence[str]] | None = None,
        window_end: datetime,
        window_months: int = 18,
        severities: Sequence[str] = ("CRITICAL", "HIGH"),
        recent_limit: int = 5000,
    ) -> None:
        self._client = client
        self._attack_links = dict(attack_links or {})
        self._window_end = window_end
        self._window_months = window_months
        self._severities = list(severities)
        self._recent_limit = recent_limit

    def _select(self, *, force: bool = False) -> Iterator[dict[str, Any]]:
        """Every selected record, ATT&CK-linked first, then newest-first by window.

        Ordering is not cosmetic. The guaranteed-relevant CVEs are fetched
        before any capped slice, so lowering the cap can never cost the corpus
        the documents it most needs, and a fetch interrupted halfway still
        leaves a coherent subset rather than an arbitrary one.
        """
        seen: set[str] = set()

        for cve_id in sorted(self._attack_links):
            record = self._client.get_cve(cve_id, force=force)
            # ATT&CK cites ids NVD has rejected or never held; a miss is data.
            if record is None or is_rejected(record):
                continue
            seen.add(str(record.get("id", "")).upper())
            yield record

        taken = 0
        for start, end in windows_back_from(self._window_end, self._window_months):
            for severity in self._severities:
                if taken >= self._recent_limit:
                    return
                for record in self._client.iter_window(
                    start,
                    end,
                    severity=severity,
                    limit=self._recent_limit - taken,
                    force=force,
                ):
                    cve_id = str(record.get("id", "")).upper()
                    if cve_id in seen or is_rejected(record):
                        continue
                    seen.add(cve_id)
                    taken += 1
                    yield record

    def fetch(self, *, force: bool = False) -> int:
        """Populate the cache. Returns how many records were selected.

        ``force`` re-requests everything instead of reading the cache, which is
        how a stale corpus is refreshed. It costs the full request budget, so
        it is never the default.
        """
        return sum(1 for _ in self._select(force=force))

    def load(self) -> Iterator[Document]:
        """Re-runs selection against the cache, so this costs no requests."""
        for record in self._select():
            cve_id = str(record.get("id", "")).upper()
            yield to_document(record, self._attack_links.get(cve_id, ()))
