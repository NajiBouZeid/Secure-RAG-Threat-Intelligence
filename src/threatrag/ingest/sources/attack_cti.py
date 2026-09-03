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
import re
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


# Actors whose ``uses`` edges become a technique's procedure examples. Campaigns
# are included even though they are not indexed as documents of their own: the
# procedure text is what matters, not the campaign entry.
PROCEDURE_ACTORS = frozenset({"intrusion-set", "malware", "tool", "campaign"})

# ATT&CK does not record CVEs in a structured field. They appear only in prose,
# so the CVE-to-technique link has to be read out of the text -- see
# :meth:`AttackCtiSource.cve_mentions` for what that costs.
CVE_PATTERN = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)


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


def _log_sources(analytic: dict[str, Any]) -> list[str]:
    """Channels an analytic reads, e.g. ``WinEventLog:Security (EventCode=4624)``."""
    out: list[str] = []
    for ref in analytic.get("x_mitre_log_source_references", []):
        name = str(ref.get("name", "")).strip()
        if not name:
            continue
        channel = str(ref.get("channel", "")).strip()
        out.append(f"{name} ({channel})" if channel else name)
    return out


class AttackCtiSource:
    """Normalises the ATT&CK Enterprise bundle into :class:`Document` objects."""

    name = "attack_cti"
    source_type = SourceType.ATTACK_CTI

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

    def _render_strategy(self, strategy: dict[str, Any], by_stix: dict[str, dict[str, Any]]) -> str:
        lines: list[str] = [f"### {strategy.get('name', 'Detection strategy')}"]
        for ref in _as_str_list(strategy.get("x_mitre_analytic_refs")):
            analytic = by_stix.get(ref)
            if analytic is None:
                continue
            if analytic.get("x_mitre_deprecated") and not self._include_deprecated:
                continue
            description = str(analytic.get("description", "")).strip()
            if description:
                lines.append(description)
            sources = _log_sources(analytic)
            if sources:
                lines.append("Log sources: " + "; ".join(sources))
        return "\n".join(lines) + "\n" if len(lines) > 1 else ""

    def _detection_index(self, objects: list[dict[str, Any]]) -> dict[str, str]:
        """Map technique STIX id -> rendered detection guidance.

        ATT&CK v18 moved detection off the technique: the ``x_mitre_detection``
        string is gone and the guidance now lives in standalone
        ``x-mitre-detection-strategy`` objects that point back through a
        ``detects`` relationship, each fanning out to analytics that carry the
        concrete log sources. Rebuilding the section from that join is what keeps
        the Detection block populated -- without it every technique in a current
        bundle indexes with its description alone, the structural chunker loses
        one of its three sections, and detection-shaped queries retrieve nothing.
        """
        by_stix = {obj["id"]: obj for obj in objects if "id" in obj}
        blocks: dict[str, list[str]] = {}
        for rel in objects:
            if rel.get("type") != "relationship" or rel.get("relationship_type") != "detects":
                continue
            strategy = by_stix.get(str(rel.get("source_ref", "")))
            if strategy is None or strategy.get("type") != "x-mitre-detection-strategy":
                continue
            if strategy.get("revoked") and not self._include_revoked:
                continue
            if strategy.get("x_mitre_deprecated") and not self._include_deprecated:
                continue
            rendered = self._render_strategy(strategy, by_stix)
            if rendered:
                blocks.setdefault(str(rel.get("target_ref", "")), []).append(rendered)
        return {key: "\n".join(value) for key, value in blocks.items()}

    def _procedure_index(self, objects: list[dict[str, Any]]) -> dict[str, str]:
        """Map technique STIX id -> the procedure examples ATT&CK lists for it.

        Without this the corpus cannot answer the question the gold set asks.
        Which techniques a group uses is recorded only as a STIX ``uses`` edge;
        the group document is a description and a list of aliases and names not
        one technique. Measured against a corpus that omits these edges, only 5
        of 941 gold (group, technique) pairs had their evidence indexed at all,
        so Recall@k scored chance no matter how good the retriever was.

        Every ``uses`` edge carries a written description naming the actor --
        the same text ATT&CK renders as "Procedure Examples" on a technique
        page -- so folding it into the technique document restores the evidence
        without inventing anything or weakening MITRE's curated labels.
        """
        by_stix = {obj["id"]: obj for obj in objects if "id" in obj}
        entries: dict[str, list[str]] = {}
        for rel in objects:
            if rel.get("type") != "relationship" or rel.get("relationship_type") != "uses":
                continue
            description = str(rel.get("description", "")).strip()
            if not description:
                continue
            actor = by_stix.get(str(rel.get("source_ref", "")))
            target = by_stix.get(str(rel.get("target_ref", "")))
            if actor is None or target is None or target.get("type") != "attack-pattern":
                continue
            if actor.get("type") not in PROCEDURE_ACTORS:
                continue
            if actor.get("revoked") and not self._include_revoked:
                continue
            if actor.get("x_mitre_deprecated") and not self._include_deprecated:
                continue
            actor_id = _external_id(actor)
            actor_name = str(actor.get("name", actor_id or "Unknown"))
            label = f"{actor_name} ({actor_id})" if actor_id else actor_name
            entries.setdefault(str(rel["target_ref"]), []).append(f"- {label}: {description}")
        return {key: "\n".join(value) for key, value in entries.items()}

    def load(self) -> Iterable[Document]:
        objects = self._bundle()
        detection = self._detection_index(objects)
        procedures = self._procedure_index(objects)
        for obj in objects:
            if self._keep(obj):
                stix_id = str(obj.get("id", ""))
                yield self._to_document(
                    obj, detection.get(stix_id, ""), procedures.get(stix_id, "")
                )

    def _to_document(
        self, obj: dict[str, Any], detection: str = "", procedures: str = ""
    ) -> Document:
        kind = STIX_KINDS[obj["type"]]
        attack_id = _external_id(obj) or str(obj["id"])
        name = str(obj.get("name", attack_id))
        tactics = _tactics(obj)
        platforms = _as_str_list(obj.get("x_mitre_platforms"))
        aliases = [alias for alias in _as_str_list(obj.get("aliases")) if alias != name]

        body = "".join(
            (
                section("Description", obj.get("description")),
                # Older bundles still carry the inline string; prefer it when present.
                section("Detection", str(obj.get("x_mitre_detection") or "") or detection),
                section("Tactics", ", ".join(tactics)),
                section("Platforms", ", ".join(platforms)),
                section("Aliases", ", ".join(aliases)),
                # Last: it is the longest section and the least useful in isolation.
                section("Procedure Examples", procedures),
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

    def cve_mentions(self) -> dict[str, list[str]]:
        """CVE id -> the ATT&CK ids whose text mentions it, sorted.

        This is the join that makes the CVE corpus worth having: without it the
        CVEs sit beside ATT&CK in the index sharing no vocabulary, and a
        question about a vulnerability retrieves nothing about how it is
        actually used. It also picks the CVEs that are guaranteed relevant to
        this corpus, so a bounded fetch spends its budget well.

        **The link is read out of prose, and that is a real limitation rather
        than an implementation detail.** ATT&CK has no structured CVE field:
        every mention is inside a description, so this is a regular expression
        over free text. A CVE named only as counter-example still links, and a
        vulnerability described in words rather than by id does not link at all.
        The claim the metadata can support is "ATT&CK discusses this CVE here",
        not "this technique exploits this CVE" -- so it belongs in metadata as a
        navigational cross-reference, and must not be promoted into gold labels
        the way MITRE's curated ``uses`` edges are in
        :meth:`group_technique_pairs`.

        Only ``description`` fields are searched, matching what is actually
        indexed. External reference titles also mention CVEs, but those are
        citation metadata this project never puts in a document, and a CVE in
        the title of a linked blog post is a much weaker signal than one in
        MITRE's own prose.
        """
        objects = self._bundle()
        by_stix = {obj["id"]: obj for obj in objects if "id" in obj}
        mentions: dict[str, set[str]] = {}

        def record(cve_ids: set[str], holder: dict[str, Any]) -> None:
            if holder.get("revoked") and not self._include_revoked:
                return
            if holder.get("x_mitre_deprecated") and not self._include_deprecated:
                return
            attack_id = _external_id(holder)
            if attack_id is None:
                return
            for cve_id in cve_ids:
                mentions.setdefault(cve_id, set()).add(attack_id)

        for obj in objects:
            found = {
                match.group(0).upper()
                for match in CVE_PATTERN.finditer(str(obj.get("description") or ""))
            }
            if not found:
                continue
            if obj.get("type") == "relationship":
                # A relationship carries no id of its own. Its description is
                # the procedure text folded into the technique document, so the
                # mention belongs to both ends of the edge.
                for ref in ("source_ref", "target_ref"):
                    endpoint = by_stix.get(str(obj.get(ref, "")))
                    if endpoint is not None:
                        record(found, endpoint)
            else:
                record(found, obj)

        return {cve_id: sorted(ids) for cve_id, ids in sorted(mentions.items())}

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
