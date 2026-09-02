"""MITRE ATT&CK Enterprise (STIX 2.1) source.

Pulls the single ``enterprise-attack.json`` bundle rather than cloning
``mitre/cti``: one ~40 MB file instead of a few hundred MB of git history, and
it resumes cleanly on a bad connection.

Beyond documents, this module also derives the retrieval gold set from STIX
``uses`` relationships. That matters, because the ground truth is then MITRE's
own curated group-to-technique mapping -- Recall@k is measured against real
labels rather than against questions an LLM invented about the very corpus it
is being tested on.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from threatrag.domain.models import TLP, Document, SourceType, TrustTier
from threatrag.ingest.sources.base import download, section

ATTACK_SOURCE_NAME = "mitre-attack"

# STIX type -> the label used in document titles and metadata.
STIX_KINDS: dict[str, str] = {
    "attack-pattern": "technique",
    "intrusion-set": "group",
    "malware": "software",
    "tool": "software",
    "course-of-action": "mitigation",
    "x-mitre-tactic": "tactic",
}


def _external_id(obj: dict[str, Any]) -> str | None:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == ATTACK_SOURCE_NAME and ref.get("external_id"):
            return str(ref["external_id"])
    return None


def _external_url(obj: dict[str, Any]) -> str | None:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == ATTACK_SOURCE_NAME:
            url = ref.get("url")
            return str(url) if url else None
    return None


def _tactics(obj: dict[str, Any]) -> list[str]:
    return [
        phase["phase_name"]
        for phase in obj.get("kill_chain_phases", [])
        if phase.get("kill_chain_name") == "mitre-attack" and phase.get("phase_name")
    ]


def _as_str_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


class AttackCtiSource:
    """Normalises the ATT&CK Enterprise bundle into :class:`Document` objects."""

    name = "attack_cti"

    def __init__(
        self,
        raw_dir: Path,
        url: str,
        *,
        include_revoked: bool = False,
        include_deprecated: bool = False,
    ) -> None:
        self._url = url
        self._path = raw_dir / "enterprise-attack.json"
        self._include_revoked = include_revoked
        self._include_deprecated = include_deprecated

    @property
    def path(self) -> Path:
        return self._path

    def fetch(self, *, force: bool = False) -> None:
        download(self._url, self._path, force=force)

    def _bundle(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            raise FileNotFoundError(
                f"{self._path} missing. Run `threatrag fetch attack` before ingesting."
            )
        with self._path.open(encoding="utf-8") as handle:
            return list(json.load(handle).get("objects", []))

    def _keep(self, obj: dict[str, Any]) -> bool:
        if obj.get("type") not in STIX_KINDS:
            return False
        if obj.get("revoked") and not self._include_revoked:
            return False
        if obj.get("x_mitre_deprecated") and not self._include_deprecated:
            return False
        return _external_id(obj) is not None

    def load(self) -> Iterable[Document]:
        for obj in self._bundle():
            if self._keep(obj):
                yield self._to_document(obj)

    def _to_document(self, obj: dict[str, Any]) -> Document:
        kind = STIX_KINDS[obj["type"]]
        attack_id = _external_id(obj) or str(obj["id"])
        name = str(obj.get("name", attack_id))
        tactics = _tactics(obj)
        platforms = _as_str_list(obj.get("x_mitre_platforms"))
        aliases = [alias for alias in _as_str_list(obj.get("aliases")) if alias != name]

        body = "".join(
            (
                section("Description", obj.get("description")),
                section("Detection", obj.get("x_mitre_detection")),
                section("Tactics", ", ".join(tactics)),
                section("Platforms", ", ".join(platforms)),
                section("Aliases", ", ".join(aliases)),
            )
        )

        return Document(
            id=f"attack:{attack_id}",
            title=f"{attack_id} {name} ({kind})",
            text=f"# {attack_id} - {name}\n\n{body}".rstrip() + "\n",
            source_type=SourceType.ATTACK_CTI,
            source_ref=attack_id,
            url=_external_url(obj),
            # ATT&CK is public and curated: readable by everyone, and trusted
            # enough that its content may legitimately shape an answer.
            tlp=TLP.CLEAR,
            trust_tier=TrustTier.AUTHORITATIVE,
            metadata={
                "kind": kind,
                "name": name,
                "stix_id": str(obj["id"]),
                "tactics": tactics,
                "platforms": platforms,
                "aliases": aliases,
                "is_subtechnique": str(bool(obj.get("x_mitre_is_subtechnique", False))),
            },
        )

    def group_technique_pairs(self) -> dict[str, dict[str, Any]]:
        """Gold labels: for each threat group, the techniques ATT&CK says it uses.

        Returns ``{group_attack_id: {"name": str, "techniques": {tactic: [ids]}}}``.
        """
        objects = self._bundle()
        by_stix: dict[str, dict[str, Any]] = {obj["id"]: obj for obj in objects if "id" in obj}

        groups: dict[str, dict[str, Any]] = {}
        for rel in objects:
            if rel.get("type") != "relationship" or rel.get("relationship_type") != "uses":
                continue
            source = by_stix.get(str(rel.get("source_ref", "")))
            target = by_stix.get(str(rel.get("target_ref", "")))
            if source is None or target is None:
                continue
            if source.get("type") != "intrusion-set" or target.get("type") != "attack-pattern":
                continue
            if not self._keep(source) or not self._keep(target):
                continue

            group_id = _external_id(source)
            technique_id = _external_id(target)
            if group_id is None or technique_id is None:
                continue

            entry = groups.setdefault(
                group_id, {"name": str(source.get("name", group_id)), "techniques": {}}
            )
            for tactic in _tactics(target) or ["unspecified"]:
                bucket: list[str] = entry["techniques"].setdefault(tactic, [])
                if technique_id not in bucket:
                    bucket.append(technique_id)

        return groups
