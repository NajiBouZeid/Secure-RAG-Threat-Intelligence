"""Tests for the vendor report corpus.

The cleaning tests matter more than they look. A PDF that extracts badly does
not raise; it yields plausible text that embeds and retrieves, so the failure
shows up as answers quietly built on furniture and hyphen-split words. These
assertions are the only place that failure is visible.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from threatrag.domain.models import TLP, SourceType, TrustTier
from threatrag.ingest.sources.vendor_report import (
    DEFAULT_MANIFEST,
    MIN_CHARS_PER_PAGE,
    ExtractionError,
    ReportSpec,
    VendorReportSource,
    clean,
    load_manifest,
    sha256_of,
    to_document,
)

SPEC = ReportSpec(
    id="RC-TDR-2026",
    title="Threat Detection Report 2026",
    publisher="Red Canary",
    published="2026-03",
    url="https://example.test/report.pdf",
    filename="report.pdf",
)


def write_manifest(tmp_path: Path, reports: list[dict[str, object]]) -> Path:
    path = tmp_path / "vendor_reports.yaml"
    path.write_text(yaml.safe_dump({"reports": reports}), encoding="utf-8")
    return path


def entry(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "RC-TDR-2026",
        "title": "Threat Detection Report 2026",
        "publisher": "Red Canary",
        "published": "2026-03",
        "url": "https://example.test/report.pdf",
        "filename": "report.pdf",
    }
    base.update(overrides)
    return base


# -- cleaning -----------------------------------------------------------


def test_running_heads_are_stripped() -> None:
    """Furniture repeats on every page, so it would become the corpus's commonest text."""
    pages = [
        "2026 Threat Detection Report\nCloud accounts ranked first this year.\n12",
        "2026 Threat Detection Report\nEmail forwarding rules rose sharply.\n13",
        "2026 Threat Detection Report\nInfostealers continued to surge.\n14",
    ]

    body = clean(pages)

    assert body.count("2026 Threat Detection Report") == 0
    assert "Cloud accounts ranked first this year." in body
    assert "Infostealers continued to surge." in body


def test_page_numbers_are_dropped() -> None:
    pages = ["Real content here.\n7", "More content.\nPage 8 of 40"]

    body = clean(pages)

    assert "Page 8 of 40" not in body
    assert "\n7" not in body
    assert "Real content here." in body


def test_hyphenated_line_breaks_are_rejoined() -> None:
    """Otherwise the indexed vocabulary differs from the vocabulary questions use."""
    body = clean(["Adversaries used cre-\ndential dumping widely."])

    assert "credential dumping" in body
    assert "cre-" not in body


def test_a_phrase_repeated_within_one_page_is_not_furniture() -> None:
    pages = ["ransomware\nransomware\nransomware\nUnique line one.", "Unique line two."]

    body = clean(pages)

    assert "ransomware" in body


def test_blank_runs_are_collapsed() -> None:
    assert "\n\n\n" not in clean(["A line.\n\n\n\n\nAnother line.", "Second page content."])


# -- manifest -----------------------------------------------------------


def test_manifest_round_trips(tmp_path: Path) -> None:
    (spec,) = load_manifest(write_manifest(tmp_path, [entry()]))

    assert spec.id == "RC-TDR-2026"
    assert spec.via == "publisher"
    assert spec.sha256 is None
    assert spec.citation_title == "Red Canary: Threat Detection Report 2026"


def test_a_missing_field_names_the_report(tmp_path: Path) -> None:
    broken = entry()
    del broken["url"]

    with pytest.raises(ValueError, match="RC-TDR-2026"):
        load_manifest(write_manifest(tmp_path, [broken]))


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        load_manifest(write_manifest(tmp_path, [entry(), entry(filename="other.pdf")]))


def test_the_committed_manifest_is_valid() -> None:
    """The manifest ships with the repo, so a typo in it is a broken clone."""
    specs = load_manifest(DEFAULT_MANIFEST)

    assert len(specs) == 10
    assert all(spec.url.startswith("https://") for spec in specs)
    assert all(spec.filename.endswith(".pdf") for spec in specs)
    assert {spec.via for spec in specs} <= {"publisher", "mirror"}
    # Distinct filenames, or one report would overwrite another on disk.
    assert len({spec.filename for spec in specs}) == len(specs)


def test_the_manifest_covers_more_than_one_publisher() -> None:
    """A single-vendor corpus would measure one house style, not vendor tier."""
    publishers = {spec.publisher for spec in load_manifest(DEFAULT_MANIFEST)}

    assert len(publishers) >= 5


# -- documents ----------------------------------------------------------


def test_documents_are_vendor_tier_but_publicly_readable() -> None:
    """The distinction the two axes exist for, and the precondition for M4."""
    document = to_document(SPEC, "Body text about adversary tradecraft.")

    assert document.trust_tier is TrustTier.VENDOR
    assert document.tlp is TLP.CLEAR
    assert document.source_type is SourceType.VENDOR_REPORT
    assert document.source_ref == "RC-TDR-2026"
    assert document.url == "https://example.test/report.pdf"


def test_document_records_provenance() -> None:
    mirrored = to_document(replace(SPEC, via="mirror"), "Body.")

    assert mirrored.metadata["provenance"] == "mirror"
    assert to_document(SPEC, "Body.").metadata["provenance"] == "publisher"


def test_document_keeps_publisher_and_date_in_the_text() -> None:
    text = to_document(SPEC, "Body text.").text

    assert "Red Canary, 2026-03" in text
    assert "Body text." in text


# -- fetch and load -----------------------------------------------------


def test_a_hash_mismatch_fails_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A URL that rots into a login page must not be ingested as intelligence."""
    manifest = write_manifest(tmp_path, [entry(sha256="0" * 64)])
    source = VendorReportSource(tmp_path, manifest)

    def fake_download(url: str, destination: Path, *, force: bool = False) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"not the expected document")
        return destination

    monkeypatch.setattr("threatrag.ingest.sources.vendor_report.download", fake_download)

    with pytest.raises(ValueError, match="sha256 mismatch"):
        source.fetch()


def test_a_matching_hash_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"the expected document"
    digest = __import__("hashlib").sha256(payload).hexdigest()
    manifest = write_manifest(tmp_path, [entry(sha256=digest)])
    source = VendorReportSource(tmp_path, manifest)

    def fake_download(url: str, destination: Path, *, force: bool = False) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return destination

    monkeypatch.setattr("threatrag.ingest.sources.vendor_report.download", fake_download)
    source.fetch()

    assert source.observed["RC-TDR-2026"] == digest


def test_loading_before_fetching_says_what_to_run(tmp_path: Path) -> None:
    source = VendorReportSource(tmp_path, write_manifest(tmp_path, [entry()]))

    with pytest.raises(FileNotFoundError, match="threatrag fetch vendor"):
        list(source.load())


def test_sha256_of_a_file(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")

    assert sha256_of(path) == __import__("hashlib").sha256(b"abc").hexdigest()


def test_extraction_floor_is_a_real_guard() -> None:
    """A scanned PDF yields a few characters per page that still embed and retrieve."""
    assert MIN_CHARS_PER_PAGE >= 100
    assert ExtractionError.__mro__[1] is RuntimeError


def test_downloads_identify_the_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Some vendor CDNs 403 httpx's default UA while serving the same public PDF."""
    from threatrag.ingest.sources import base

    seen: dict[str, str] = {}

    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self, chunk_size: int = 0) -> list[bytes]:
            return [b"%PDF-1.7"]

    def fake_stream(method: str, url: str, **kwargs: object) -> FakeStream:
        seen.update(kwargs.get("headers") or {})  # type: ignore[arg-type]
        return FakeStream()

    monkeypatch.setattr(base.httpx, "stream", fake_stream)
    base.download("https://example.test/x.pdf", tmp_path / "x.pdf")

    assert seen["User-Agent"] == base.USER_AGENT
    assert "httpx" not in seen["User-Agent"]


def test_doubled_heading_characters_are_collapsed() -> None:
    """pdfplumber renders faux-bold twice: "THREAT" arrives as "TTHHRREEAATT"."""
    body = clean(
        ["TTHHRREEAATT DDEETTEECCTTIIOONN\nOrdinary prose follows here.", "Page two text."]
    )

    assert "THREAT DETECTION" in body
    assert "TTHHRREEAATT" not in body


def test_ordinary_prose_with_double_letters_survives() -> None:
    """The collapse must never touch "coffee", "HTTP" or "successfully"."""
    line = "The committee successfully assessed all HTTP traffic."
    body = clean([line, "Second page."])

    assert line in body


def test_table_of_contents_leaders_are_dropped() -> None:
    pages = ["Email threats ....................... 26\nReal analysis begins here.", "Page two."]

    body = clean(pages)

    assert "Email threats" not in body
    assert "Real analysis begins here." in body


def test_footers_differing_only_by_page_number_are_furniture() -> None:
    """Exact matching misses these, because the number changes on every page."""
    pages = [
        "Content one.\n(c) 2026 Cisco and/or its affiliates. talosintelligence.com page 2",
        "Content two.\n(c) 2026 Cisco and/or its affiliates. talosintelligence.com page 3",
        "Content three.\n(c) 2026 Cisco and/or its affiliates. talosintelligence.com page 4",
    ]

    body = clean(pages)

    assert "talosintelligence.com" not in body
    assert "Content two." in body
