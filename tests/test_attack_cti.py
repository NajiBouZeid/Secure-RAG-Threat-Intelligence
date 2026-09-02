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
