"""Vendor threat reports: published PDFs as VENDOR-tier documents.

This is the corpus that makes trust tier do something. Until now every indexed
document was AUTHORITATIVE -- MITRE's own text, NVD's records, and internal
notes written in this repository -- so the tier field was carried faithfully
through the pipeline and never distinguished anything. Vendor reports are
public, useful and *less authoritative than MITRE*, which is exactly the
distinction the two axes exist to express, and it is the precondition for the
Phase 2 injection attack: attacker-influenced text sitting beside MITRE's with
nothing in the index telling them apart.

**Extraction quality is a security property here, not a nicety.** A PDF that
extracts badly does not fail loudly; it produces plausible-looking garbage that
embeds, indexes and retrieves, and then silently poisons answers with text
nobody wrote. That is why the reports are hand-picked rather than scraped, why
:func:`extract` refuses a document whose yield per page is implausible, and why
the manifest records a hash: a URL that rots into a login page would otherwise
be ingested as though it were intelligence.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml

from threatrag.domain.models import TLP, Document, SourceType, TrustTier
from threatrag.ingest.sources.base import download

DEFAULT_MANIFEST = Path("corpora/vendor_reports.yaml")

# A text-bearing report page carries far more than this. Anything below it means
# a scanned image, an extraction failure, or a login page wearing a .pdf suffix.
MIN_CHARS_PER_PAGE = 200

# A line repeated on this share of pages is running-head furniture, not content.
_FURNITURE_THRESHOLD = 0.5

_PAGE_NUMBER = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.IGNORECASE)
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_BLANK_RUN = re.compile(r"\n{3,}")


class ExtractionError(RuntimeError):
    """A PDF produced too little text to be trusted as a source document."""


@dataclass(frozen=True, slots=True)
class ReportSpec:
    """One entry in the manifest."""

    id: str
    title: str
    publisher: str
    published: str
    url: str
    filename: str
    via: str = "publisher"
    sha256: str | None = None

    @property
    def citation_title(self) -> str:
        return f"{self.publisher}: {self.title}"


def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> list[ReportSpec]:
    manifest = Path(path)
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} missing; expected it in the repository.")
    with manifest.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    entries = payload.get("reports") or []
    if not isinstance(entries, list):
        raise ValueError(f"{manifest}: 'reports' must be a list")

    specs: list[ReportSpec] = []
    seen: set[str] = set()
    for entry in entries:
        try:
            spec = ReportSpec(
                id=str(entry["id"]),
                title=str(entry["title"]),
                publisher=str(entry["publisher"]),
                published=str(entry.get("published", "")),
                url=str(entry["url"]),
                filename=str(entry["filename"]),
                via=str(entry.get("via", "publisher")),
                sha256=str(entry["sha256"]) if entry.get("sha256") else None,
            )
        except KeyError as exc:
            raise ValueError(f"Vendor report {entry.get('id', '?')} is missing {exc}") from exc
        if spec.id in seen:
            raise ValueError(f"Duplicate vendor report id {spec.id!r}")
        seen.add(spec.id)
        specs.append(spec)
    return specs


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _strip_furniture(pages: list[str]) -> list[str]:
    """Remove running heads, footers and page numbers.

    They repeat on every page, so left in they become the most frequent text in
    the corpus: a chunk of pure furniture retrieves for anything and answers
    nothing. Detection is by repetition rather than by position, because vendor
    reports put their furniture in wildly different places.
    """
    counts: Counter[str] = Counter()
    for page in pages:
        # Count each distinct line once per page, so a phrase legitimately
        # repeated within one page is not mistaken for a running head.
        counts.update({line.strip() for line in page.splitlines() if line.strip()})

    threshold = max(2, int(len(pages) * _FURNITURE_THRESHOLD))
    furniture = {line for line, count in counts.items() if count >= threshold}

    cleaned: list[str] = []
    for page in pages:
        kept = [
            line
            for line in page.splitlines()
            if line.strip()
            and line.strip() not in furniture
            and not _PAGE_NUMBER.match(line.strip())
        ]
        cleaned.append("\n".join(kept))
    return cleaned


def clean(pages: list[str]) -> str:
    """Join extracted pages into one body of readable prose."""
    text = "\n\n".join(page for page in _strip_furniture(pages) if page.strip())
    # Rejoin words split across a line break, or the vocabulary the retriever
    # matches on quietly differs from the vocabulary a question uses.
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


def extract(path: Path) -> str:
    """Pull cleaned text out of a PDF, refusing an implausible yield.

    Raises rather than returning something usable-looking, because the failure
    this guards against is silent: a scanned or gated PDF yields a handful of
    characters per page that still embed and still retrieve.
    """
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - depends on the pdf extra
        raise ImportError("Vendor reports need the pdf extra: pip install -e '.[pdf]'") from exc

    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")

    if not pages:
        raise ExtractionError(f"{path.name}: no pages found")

    body = clean(pages)
    per_page = len(body) / len(pages)
    if per_page < MIN_CHARS_PER_PAGE:
        raise ExtractionError(
            f"{path.name}: {per_page:.0f} chars/page over {len(pages)} pages, below the "
            f"{MIN_CHARS_PER_PAGE} minimum. Likely a scanned or gated PDF; ingesting it "
            "would put plausible-looking noise into the index."
        )
    return body


def to_document(spec: ReportSpec, body: str) -> Document:
    return Document(
        id=f"vendor:{spec.id}",
        title=spec.citation_title,
        text=f"# {spec.title}\n\n{spec.publisher}, {spec.published}\n\n{body}".rstrip() + "\n",
        source_type=SourceType.VENDOR_REPORT,
        source_ref=spec.id,
        url=spec.url,
        # Published openly, so everyone may read it -- but a vendor's analysis
        # is not MITRE's curated reference, and the tier says so. This is the
        # first non-AUTHORITATIVE public content in the index.
        tlp=TLP.CLEAR,
        trust_tier=TrustTier.VENDOR,
        metadata={
            "publisher": spec.publisher,
            "published": spec.published,
            "provenance": spec.via,
        },
    )


class VendorReportSource:
    """Fetches the manifest's PDFs and turns them into documents."""

    name = "vendor_report"
    source_type = SourceType.VENDOR_REPORT

    def __init__(
        self,
        raw_dir: Path,
        manifest: Path | str = DEFAULT_MANIFEST,
    ) -> None:
        self._dir = Path(raw_dir) / "vendor"
        self._specs = load_manifest(manifest)
        # Hashes observed on the last fetch, for reporting back into the
        # manifest the first time a report is downloaded.
        self.observed: dict[str, str] = {}

    @property
    def specs(self) -> list[ReportSpec]:
        return list(self._specs)

    def path_for(self, spec: ReportSpec) -> Path:
        return self._dir / spec.filename

    def fetch(self, *, force: bool = False) -> None:
        for spec in self._specs:
            path = self.path_for(spec)
            download(spec.url, path, force=force)
            digest = sha256_of(path)
            self.observed[spec.id] = digest
            if spec.sha256 and digest != spec.sha256:
                raise ValueError(
                    f"{spec.id}: sha256 mismatch. The manifest expects {spec.sha256[:12]}... "
                    f"and the download is {digest[:12]}.... The URL may have rotted into a "
                    "different document; verify it before trusting the content."
                )

    def load(self) -> Iterator[Document]:
        for spec in self._specs:
            path = self.path_for(spec)
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} missing. Run `threatrag fetch vendor` before ingesting."
                )
            yield to_document(spec, extract(path))
