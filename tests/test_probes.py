"""Identifier probe construction."""

from __future__ import annotations

from threatrag.eval.probes import build_identifier_probes

DOCS = [
    ("T1127.003", "T1127.003 JamPlus (technique)"),
    ("T1106", "T1106 Native API (technique)"),
    ("G1024", "G1024 Akira (group)"),
    ("S1129", "S1129 Akira (software)"),
    ("G0007", "G0007 APT28 (group)"),
    ("M1029", "M1029 Remote Data Storage (mitigation)"),
    ("CVE-2026-41397", "CVE-2026-41397 (HIGH)"),
    ("INT-2026-009", "INT-2026-009 Forum paste alleging access"),
]


def _by_id(docs: list[tuple[str, str]] = DOCS) -> dict[str, dict[str, str]]:
    return {q.id: {"question": q.question, **q.tags} for q in build_identifier_probes(docs).queries}


def test_attack_documents_get_an_id_probe_and_a_name_twin() -> None:
    probes = _by_id()

    assert probes["T1127.003:id"]["question"] == "What is T1127.003?"
    assert probes["T1127.003:name"]["question"] == "What is JamPlus?"
    assert probes["T1127.003:name"]["kind"] == "technique"


def test_a_name_shared_by_two_documents_is_dropped_from_both() -> None:
    """Its twin would have two right answers and be scored against one."""
    probes = _by_id()

    assert not any(key.startswith(("G1024", "S1129")) for key in probes)
    assert "G0007:name" in probes


def test_cves_are_identifier_only() -> None:
    probes = _by_id()

    assert probes["CVE-2026-41397:id"]["kind"] == "cve"
    assert "CVE-2026-41397:name" not in probes


def test_other_documents_are_not_probed() -> None:
    probes = _by_id()

    assert not any(key.startswith(("M1029", "INT-")) for key in probes)


def test_each_probe_is_relevant_only_to_its_own_document() -> None:
    for query in build_identifier_probes(DOCS).queries:
        assert query.relevant == [query.id.split(":")[0]]


def test_sampling_is_seeded_and_capped_per_kind() -> None:
    docs = [(f"T{1000 + i}", f"T{1000 + i} Technique {i} (technique)") for i in range(40)]

    first = build_identifier_probes(docs, per_kind=10)
    second = build_identifier_probes(docs, per_kind=10)

    assert [q.id for q in first.queries] == [q.id for q in second.queries]
    assert len(first.queries) == 20
