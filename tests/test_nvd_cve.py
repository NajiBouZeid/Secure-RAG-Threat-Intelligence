"""Tests for CVE document shaping and selection.

The shaping tests lean on the terms-of-use commitment made in the module
docstring: the description is reproduced verbatim, and every truncation is
visible. Those are the assertions that keep "we reformatted it for retrieval"
an honest description of what this code does.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from threatrag.domain.models import TLP, SourceType, TrustTier
from threatrag.ingest.sources.nvd_api import NvdClient
from threatrag.ingest.sources.nvd_cve import (
    MAX_PRODUCTS,
    NvdCveSource,
    is_rejected,
    to_document,
)

DESCRIPTION = (
    "Malicious code was discovered in the upstream tarballs of xz, starting "
    "with version 5.6.0. Through a series of complex obfuscations, the build "
    "process extracts a prebuilt object file which modifies functions in liblzma."
)


def record(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "CVE-2024-3094",
        "published": "2024-03-29T17:15:21.150",
        "lastModified": "2024-04-19T13:15:07.000",
        "vulnStatus": "Analyzed",
        "descriptions": [
            {"lang": "es", "value": "Se descubrio codigo malicioso."},
            {"lang": "en", "value": DESCRIPTION},
        ],
        "metrics": {
            "cvssMetricV31": [
                {
                    "type": "Primary",
                    "cvssData": {
                        "version": "3.1",
                        "baseScore": 10.0,
                        "baseSeverity": "CRITICAL",
                        "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
                    },
                }
            ]
        },
        "weaknesses": [
            {"description": [{"lang": "en", "value": "CWE-506"}]},
            {"description": [{"lang": "en", "value": "NVD-CWE-noinfo"}]},
        ],
        "configurations": [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {"criteria": "cpe:2.3:a:tukaani:xz:5.6.0:*:*:*:*:*:*:*"},
                            {"criteria": "cpe:2.3:a:tukaani:xz:5.6.1:*:*:*:*:*:*:*"},
                            {"criteria": "cpe:2.3:o:red_hat:fedora:40:*:*:*:*:*:*:*"},
                        ]
                    }
                ]
            }
        ],
        "references": [
            {"url": "https://www.openwall.com/lists/oss-security/2024/03/29/4", "tags": ["Patch"]},
            {"url": "https://news.ycombinator.com/item?id=39865810"},
        ],
    }
    base.update(overrides)
    return base


# -- shaping ------------------------------------------------------------


def test_description_is_reproduced_verbatim() -> None:
    """The terms of use are only defensible if the content is not rewritten."""
    document = to_document(record())

    assert DESCRIPTION in document.text
    assert "## Description" in document.text


def test_english_description_is_preferred() -> None:
    assert "Se descubrio" not in to_document(record()).text


def test_severity_and_vector_are_rendered() -> None:
    text = to_document(record()).text

    assert "CVSS v3.1 base score 10.0 (CRITICAL)" in text
    assert "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H" in text


def test_newer_scoring_version_wins() -> None:
    """NVD carries several CVSS versions at once; a fixed key would drop scores."""
    payload = record()
    payload["metrics"]["cvssMetricV40"] = [
        {
            "type": "Primary",
            "cvssData": {"version": "4.0", "baseScore": 9.3, "baseSeverity": "CRITICAL"},
        }
    ]

    assert "CVSS v4.0 base score 9.3" in to_document(payload).text


def test_primary_metric_outranks_a_vendor_supplied_one() -> None:
    payload = record()
    payload["metrics"]["cvssMetricV31"].insert(
        0,
        {
            "type": "Secondary",
            "cvssData": {"version": "3.1", "baseScore": 4.0, "baseSeverity": "MEDIUM"},
        },
    )

    assert "base score 10.0" in to_document(payload).text


def test_a_cve_with_no_metrics_still_produces_a_document() -> None:
    document = to_document(record(metrics={}))

    assert "## Severity" not in document.text
    assert DESCRIPTION in document.text
    assert document.metadata["cvss_score"] == ""


def test_placeholder_cwes_are_dropped() -> None:
    document = to_document(record())

    assert "CWE-506" in document.text
    assert "NVD-CWE-noinfo" not in document.text
    assert document.metadata["cwes"] == ["CWE-506"]


def test_products_are_deduplicated_without_versions() -> None:
    """Version strings multiply one product into hundreds of near-identical lines."""
    text = to_document(record()).text

    assert text.count("tukaani xz") == 1
    assert "red hat fedora" in text
    assert "5.6.0" not in text.split("## Affected Products")[1]


def test_long_product_lists_are_truncated_visibly() -> None:
    """An answer built on a silently-cut list looks complete and is not."""
    many = [
        {"criteria": f"cpe:2.3:a:vendor{i}:product{i}:1.0:*:*:*:*:*:*:*"}
        for i in range(MAX_PRODUCTS + 30)
    ]
    document = to_document(record(configurations=[{"nodes": [{"cpeMatch": many}]}]))

    assert "... and 30 more affected products not shown" in document.text
    assert document.metadata["products_truncated"] == "True"
    # The description is never a casualty of bounding the lists.
    assert DESCRIPTION in document.text


def test_attack_links_are_a_cross_reference_section() -> None:
    document = to_document(record(), ["T1190", "T1203"])

    assert "## Discussed by ATT&CK" in document.text
    assert "T1190, T1203" in document.text
    assert document.metadata["attack_ids"] == ["T1190", "T1203"]


def test_document_carries_provenance_and_classification() -> None:
    document = to_document(record())

    assert document.source_ref == "CVE-2024-3094"
    assert document.source_type is SourceType.NVD_CVE
    assert document.tlp is TLP.CLEAR
    assert document.trust_tier is TrustTier.AUTHORITATIVE
    # The unmodified record stays one click away.
    assert document.url == "https://nvd.nist.gov/vuln/detail/CVE-2024-3094"


def test_rejected_entries_are_recognised() -> None:
    assert is_rejected(record(vulnStatus="Rejected"))
    assert is_rejected(
        record(descriptions=[{"lang": "en", "value": "** REJECT ** Not a vulnerability"}])
    )
    assert not is_rejected(record())


# -- selection ----------------------------------------------------------


class Fake:
    def __init__(self, pages: dict[str, dict[str, Any]]) -> None:
        self._pages = pages
        self.urls: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        cve_id = request.url.params.get("cveId")
        if cve_id:
            payload = self._pages.get(cve_id, {"vulnerabilities": [], "totalResults": 0})
            return httpx.Response(200, json=payload)
        severity = request.url.params.get("cvssV3Severity", "")
        payload = self._pages.get(f"window:{severity}", {"vulnerabilities": [], "totalResults": 0})
        return httpx.Response(200, json=payload)

    def client(self, tmp_path: Path) -> NvdClient:
        return NvdClient(
            tmp_path / "nvd",
            client=httpx.Client(transport=httpx.MockTransport(self.handler)),
            sleep=lambda _: None,
        )


def _wrap(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "totalResults": len(records),
        "vulnerabilities": [{"cve": r} for r in records],
    }


def _source(tmp_path: Path, fake: Fake, **kwargs: Any) -> NvdCveSource:
    kwargs.setdefault("window_end", datetime(2026, 9, 1))
    kwargs.setdefault("window_months", 4)
    kwargs.setdefault("severities", ["CRITICAL"])
    return NvdCveSource(fake.client(tmp_path), **kwargs)


def test_attack_linked_cves_come_first(tmp_path: Path) -> None:
    """Lowering the cap must never cost the corpus its guaranteed-relevant CVEs."""
    fake = Fake(
        {
            "CVE-2021-44228": _wrap([record(id="CVE-2021-44228")]),
            "window:CRITICAL": _wrap([record(id="CVE-2026-0001")]),
        }
    )
    source = _source(tmp_path, fake, attack_links={"CVE-2021-44228": ["T1190"]})

    ids = [d.source_ref for d in source.load()]

    assert ids[0] == "CVE-2021-44228"
    assert "CVE-2026-0001" in ids


def test_attack_links_reach_the_document(tmp_path: Path) -> None:
    fake = Fake({"CVE-2021-44228": _wrap([record(id="CVE-2021-44228")])})
    source = _source(tmp_path, fake, attack_links={"CVE-2021-44228": ["T1190"]})

    (document,) = [d for d in source.load() if d.source_ref == "CVE-2021-44228"]

    assert document.metadata["attack_ids"] == ["T1190"]


def test_a_cve_attack_cites_but_nvd_lacks_is_skipped(tmp_path: Path) -> None:
    """ATT&CK names ids NVD has rejected or never held; that is data, not failure."""
    fake = Fake({})
    source = _source(tmp_path, fake, attack_links={"CVE-1999-9999": ["T1190"]})

    assert list(source.load()) == []


def test_rejected_records_never_become_documents(tmp_path: Path) -> None:
    fake = Fake({"window:CRITICAL": _wrap([record(id="CVE-2026-0002", vulnStatus="Rejected")])})

    assert list(_source(tmp_path, fake).load()) == []


def test_the_recent_slice_is_capped(tmp_path: Path) -> None:
    many = [record(id=f"CVE-2026-{i:04d}") for i in range(50)]
    fake = Fake({"window:CRITICAL": _wrap(many)})

    documents = list(_source(tmp_path, fake, recent_limit=7).load())

    assert len(documents) == 7


def test_an_attack_linked_cve_is_not_indexed_twice(tmp_path: Path) -> None:
    duplicate = record(id="CVE-2026-0003")
    fake = Fake({"CVE-2026-0003": _wrap([duplicate]), "window:CRITICAL": _wrap([duplicate])})
    source = _source(tmp_path, fake, attack_links={"CVE-2026-0003": ["T1190"]})

    assert [d.source_ref for d in source.load()] == ["CVE-2026-0003"]


def test_load_after_fetch_costs_no_requests(tmp_path: Path) -> None:
    """Re-indexing to compare chunkers must not re-hit the API."""
    fake = Fake({"window:CRITICAL": _wrap([record(id="CVE-2026-0004")])})
    client = fake.client(tmp_path)
    source = NvdCveSource(
        client, window_end=datetime(2026, 9, 1), window_months=4, severities=["CRITICAL"]
    )

    selected = source.fetch()
    made = client.requests_made
    documents = list(source.load())

    assert selected == 1
    assert len(documents) == 1
    assert client.requests_made == made


def test_selection_is_reproducible(tmp_path: Path) -> None:
    """A window anchored to a fixed date yields the same corpus on every run."""
    fake = Fake({"window:CRITICAL": _wrap([record(id=f"CVE-2026-{i:04d}") for i in range(5)])})
    source = _source(tmp_path, fake)

    assert [d.source_ref for d in source.load()] == [d.source_ref for d in source.load()]


@pytest.mark.parametrize("severity", ["CRITICAL", "HIGH"])
def test_each_requested_severity_is_queried(tmp_path: Path, severity: str) -> None:
    fake = Fake({})
    list(_source(tmp_path, fake, severities=["CRITICAL", "HIGH"]).load())

    assert any(f"cvssV3Severity={severity}" in url for url in fake.urls)
