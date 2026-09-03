from __future__ import annotations

import json
from pathlib import Path

from threatrag.ingest.sources.attack_cti import AttackCtiSource

TECHNIQUE_STIX_ID = "attack-pattern--0001"
STRATEGY_STIX_ID = "x-mitre-detection-strategy--0001"
ANALYTIC_STIX_ID = "x-mitre-analytic--0001"


def attack_ref(external_id: str) -> dict[str, object]:
    return {
        "source_name": "mitre-attack",
        "external_id": external_id,
        "url": f"https://attack.mitre.org/techniques/{external_id}",
    }


# A minimal v18-shaped bundle: the technique carries no x_mitre_detection, and
# the guidance is reachable only through the detects relationship.
BUNDLE: dict[str, object] = {
    "type": "bundle",
    "objects": [
        {
            "id": TECHNIQUE_STIX_ID,
            "type": "attack-pattern",
            "name": "Process Injection",
            "description": "Adversaries may inject code into processes.",
            "external_references": [attack_ref("T1055")],
            "kill_chain_phases": [
                {"kill_chain_name": "mitre-attack", "phase_name": "defense-evasion"}
            ],
            "x_mitre_platforms": ["Windows"],
        },
        {
            "id": STRATEGY_STIX_ID,
            "type": "x-mitre-detection-strategy",
            "name": "Behavioural Detection of Injection",
            "x_mitre_analytic_refs": [ANALYTIC_STIX_ID],
            "external_references": [attack_ref("DET0001")],
        },
        {
            "id": ANALYTIC_STIX_ID,
            "type": "x-mitre-analytic",
            "name": "Analytic 0001",
            "description": "Detects CreateRemoteThread against a foreign process.",
            "x_mitre_log_source_references": [
                {"name": "WinEventLog:Sysmon", "channel": "EventCode=8"},
                {"name": "NSM:Flow"},
            ],
            "external_references": [attack_ref("AN0001")],
        },
        {
            "id": "relationship--0001",
            "type": "relationship",
            "relationship_type": "detects",
            "source_ref": STRATEGY_STIX_ID,
            "target_ref": TECHNIQUE_STIX_ID,
        },
    ],
}


def make_source(tmp_path: Path, bundle: dict[str, object] = BUNDLE) -> AttackCtiSource:
    (tmp_path / "enterprise-attack.json").write_text(json.dumps(bundle), encoding="utf-8")
    return AttackCtiSource(raw_dir=tmp_path, url="unused")


def test_detection_is_rebuilt_from_the_detects_relationship(tmp_path: Path) -> None:
    """ATT&CK v18 moved detection off the technique; the join must restore it.

    Without this the Detection section is empty for every technique in a current
    bundle, which costs the structural chunker one of its three sections.
    """
    (document,) = make_source(tmp_path).load()

    assert "## Detection" in document.text
    assert "Behavioural Detection of Injection" in document.text
    assert "Detects CreateRemoteThread against a foreign process." in document.text
    # Channel-qualified where a channel exists, bare where it does not.
    assert "WinEventLog:Sysmon (EventCode=8)" in document.text
    assert "NSM:Flow" in document.text


def test_inline_detection_field_still_wins_on_older_bundles(tmp_path: Path) -> None:
    """mitre/cti and archived bundles keep the flat string; it must take priority."""
    bundle = json.loads(json.dumps(BUNDLE))
    bundle["objects"][0]["x_mitre_detection"] = "Monitor WriteProcessMemory calls."

    (document,) = make_source(tmp_path, bundle).load()

    assert "Monitor WriteProcessMemory calls." in document.text
    assert "Behavioural Detection of Injection" not in document.text


def test_deprecated_strategies_are_not_indexed(tmp_path: Path) -> None:
    bundle = json.loads(json.dumps(BUNDLE))
    bundle["objects"][1]["x_mitre_deprecated"] = True

    (document,) = make_source(tmp_path, bundle).load()

    assert "## Detection" not in document.text


def test_a_strategy_with_no_usable_analytic_adds_no_empty_section(tmp_path: Path) -> None:
    """A heading alone is noise: it would dilute the embedding without adding signal."""
    bundle = json.loads(json.dumps(BUNDLE))
    bundle["objects"][2]["description"] = ""
    del bundle["objects"][2]["x_mitre_log_source_references"]

    (document,) = make_source(tmp_path, bundle).load()

    assert "## Detection" not in document.text


def test_procedure_examples_are_folded_into_the_technique(tmp_path: Path) -> None:
    """The uses edge is the only record of which group uses which technique.

    Left in the relationship graph it is unretrievable, and the gold set scores
    chance because no indexed document contains the evidence.
    """
    bundle = json.loads(json.dumps(BUNDLE))
    bundle["objects"] += [
        {
            "id": "intrusion-set--0001",
            "type": "intrusion-set",
            "name": "Ke3chang",
            "description": "A threat group.",
            "external_references": [attack_ref("G0004")],
        },
        {
            "id": "relationship--0002",
            "type": "relationship",
            "relationship_type": "uses",
            "source_ref": "intrusion-set--0001",
            "target_ref": TECHNIQUE_STIX_ID,
            "description": "Ke3chang has injected into explorer.exe.",
        },
    ]

    documents = {d.source_ref: d for d in make_source(tmp_path, bundle).load()}

    technique = documents["T1055"].text
    assert "## Procedure Examples" in technique
    assert "Ke3chang (G0004)" in technique
    assert "Ke3chang has injected into explorer.exe." in technique
    # The group keeps its own entry, and gains no procedure section of its own.
    assert "## Procedure Examples" not in documents["G0004"].text


def test_a_uses_edge_without_a_description_is_skipped(tmp_path: Path) -> None:
    bundle = json.loads(json.dumps(BUNDLE))
    bundle["objects"] += [
        {
            "id": "intrusion-set--0002",
            "type": "intrusion-set",
            "name": "Silent Group",
            "external_references": [attack_ref("G0009")],
        },
        {
            "id": "relationship--0003",
            "type": "relationship",
            "relationship_type": "uses",
            "source_ref": "intrusion-set--0002",
            "target_ref": TECHNIQUE_STIX_ID,
            "description": "   ",
        },
    ]

    documents = {d.source_ref: d for d in make_source(tmp_path, bundle).load()}

    assert "## Procedure Examples" not in documents["T1055"].text


def test_cve_mentions_link_both_ends_of_a_procedure_edge(tmp_path: Path) -> None:
    """The join that stops the CVE corpus sitting beside ATT&CK sharing no vocabulary."""
    bundle = json.loads(json.dumps(BUNDLE))
    bundle["objects"] += [
        {
            "id": "intrusion-set--0001",
            "type": "intrusion-set",
            "name": "Ke3chang",
            "description": "A threat group.",
            "external_references": [attack_ref("G0004")],
        },
        {
            "id": "relationship--0002",
            "type": "relationship",
            "relationship_type": "uses",
            "source_ref": "intrusion-set--0001",
            "target_ref": TECHNIQUE_STIX_ID,
            "description": "Ke3chang exploited cve-2021-44228 for access.",
        },
    ]

    mentions = make_source(tmp_path, bundle).cve_mentions()

    # Normalised to upper case, and attributed to the technique and the actor.
    assert mentions["CVE-2021-44228"] == ["G0004", "T1055"]


def test_cve_mentions_ignore_reference_titles(tmp_path: Path) -> None:
    """Citation metadata is never indexed, so a CVE in a blog title is not a link."""
    bundle = json.loads(json.dumps(BUNDLE))
    bundle["objects"][0]["external_references"].append(
        {"source_name": "Vendor", "description": "Analysis of CVE-2099-1234", "url": "http://x"}
    )

    assert make_source(tmp_path, bundle).cve_mentions() == {}


def test_cve_mentions_skip_deprecated_objects(tmp_path: Path) -> None:
    bundle = json.loads(json.dumps(BUNDLE))
    bundle["objects"][0]["description"] = "Exploits CVE-2021-44228."
    bundle["objects"][0]["x_mitre_deprecated"] = True

    assert make_source(tmp_path, bundle).cve_mentions() == {}
