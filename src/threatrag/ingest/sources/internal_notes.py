"""Synthetic internal threat-intelligence notes.

This is the only corpus in the project that is authored rather than fetched,
which is why it lives in ``corpora/`` under version control instead of the
gitignored ``data/`` tree.

It exists because access control was untestable without it. Every ATT&CK
document is TLP:CLEAR, so a clearance filter over the public corpus alone
returns the same results at every clearance level -- the mechanism is correct
and proves nothing. These notes give the index a real confidentiality gradient,
and later become the material the Phase 2 exfiltration and inversion attacks
try to lift out of it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from threatrag.domain.models import TLP, Document, SourceType, TrustTier

DEFAULT_PATH = Path("corpora/internal_notes.yaml")


class InternalNotesSource:
    """Loads the authored internal-notes corpus into :class:`Document` objects."""

    name = "internal_notes"
    source_type = SourceType.INTERNAL_NOTE

    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def fetch(self, *, force: bool = False) -> None:
        """No-op: this corpus ships with the repository.

        Present so the source satisfies the same protocol as the fetched ones
        and ``threatrag fetch`` does not need to special-case it.
        """

    def _raw(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            raise FileNotFoundError(f"{self._path} missing; expected it in the repository.")
        with self._path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        notes = payload.get("notes") or []
        if not isinstance(notes, list):
            raise ValueError(f"{self._path}: 'notes' must be a list")
        return [dict(note) for note in notes]

    def load(self) -> Iterator[Document]:
        seen: set[str] = set()
        for note in self._raw():
            document = self._to_document(note)
            if document.id in seen:
                raise ValueError(f"Duplicate internal note id {document.source_ref!r}")
            seen.add(document.id)
            yield document

    @staticmethod
    def _to_document(note: dict[str, Any]) -> Document:
        try:
            note_id = str(note["id"])
            title = str(note["title"])
            text = str(note["text"])
            # No defaults on the security fields. A note that forgets its TLP
            # would otherwise default to CLEAR and be published to everyone --
            # the failure mode this corpus exists to make visible.
            tlp = TLP(str(note["tlp"]))
            trust_tier = TrustTier(int(note["trust_tier"]))
        except KeyError as exc:
            raise ValueError(f"Internal note {note.get('id', '?')} is missing {exc}") from exc

        techniques = [str(item) for item in note.get("techniques", [])]
        return Document(
            id=f"internal:{note_id}",
            title=f"{note_id} {title}",
            text=f"# {note_id} - {title}\n\n{text}".rstrip() + "\n",
            source_type=SourceType.INTERNAL_NOTE,
            source_ref=note_id,
            url=None,
            tlp=tlp,
            trust_tier=trust_tier,
            metadata={"techniques": techniques, "synthetic": "true"},
        )
