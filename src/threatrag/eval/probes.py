"""Identifier probes: the questions a lexical retriever should win.

"What is T1055.001?" is the query MiniLM is weakest at and BM25 is built for, so
a probe set made of such questions favours hybrid retrieval by construction. It
is reported as its own axis for that reason and never merged into the gold-set
headline, where it would inflate a hybrid gain with questions chosen to produce
one.

Each ATT&CK probe has a twin asking for the same document by name ("What is
JamPlus?"). The pair separates two things a single probe cannot: whether a gain
comes from matching the identifier, or from the document simply being easier to
find. CVEs have no name to ask by, so their probes have no twin and are reported
as identifier-only.

Built mechanically from ``(source_ref, title)`` pairs so no question is written
from the text it is meant to retrieve. A name shared by two documents (a group
and a tool both called Akira) is dropped rather than guessed at, because its
twin would have two right answers and be scored against one.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Iterable

from threatrag.eval.goldset import GoldQuery, GoldSet

# "T1127.003 JamPlus (technique)" -> id, name, kind.
_ATTACK_TITLE = re.compile(r"^(?P<ref>\S+) (?P<name>.+) \((?P<kind>technique|group|software)\)$")
_CVE = re.compile(r"^CVE-\d{4}-\d{4,}$")

KINDS = ("technique", "group", "software", "cve")


def build_identifier_probes(
    documents: Iterable[tuple[str, str]], *, per_kind: int = 150, seed: int = 1337
) -> GoldSet:
    """Sample up to ``per_kind`` documents of each kind and ask for each by id and by name."""
    named: dict[str, list[tuple[str, str]]] = defaultdict(list)
    cves: list[str] = []
    for ref, title in sorted(set(documents)):
        match = _ATTACK_TITLE.match(title)
        if match and match["ref"] == ref:
            named[match["kind"]].append((ref, match["name"]))
        elif _CVE.match(ref):
            cves.append(ref)

    owners: dict[str, set[str]] = defaultdict(set)
    for entries in named.values():
        for ref, name in entries:
            owners[name.casefold()].add(ref)

    rng = random.Random(seed)
    queries: list[GoldQuery] = []
    for kind in ("technique", "group", "software"):
        unambiguous = [
            (ref, name) for ref, name in named[kind] if len(owners[name.casefold()]) == 1
        ]
        for ref, name in rng.sample(unambiguous, min(per_kind, len(unambiguous))):
            queries.append(_probe(ref, kind, "id", f"What is {ref}?"))
            queries.append(_probe(ref, kind, "name", f"What is {name}?"))
    for ref in rng.sample(cves, min(per_kind, len(cves))):
        queries.append(_probe(ref, "cve", "id", f"What is {ref}?"))
    return GoldSet(name="identifier-probes", queries=queries)


def _probe(ref: str, kind: str, phrasing: str, question: str) -> GoldQuery:
    return GoldQuery(
        id=f"{ref}:{phrasing}",
        question=question,
        relevant=[ref],
        tags={"kind": kind, "phrasing": phrasing},
    )
