"""BM25 sparse encoder tests.

Two properties are pinned by name because each fails silently. An identifier
split into pieces still produces matches, just the wrong ones; and a term id
that changes between processes still produces a valid query, just one that
matches nothing in an index built by an earlier run.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from threatrag.index.sparse.bm25 import Bm25Encoder, term_id, tokenize


@pytest.mark.parametrize(
    ("text", "token"),
    [
        ("uses T1055.001 for injection", "t1055.001"),
        ("see T1055", "t1055"),
        ("CVE-2024-3094 in xz", "cve-2024-3094"),
        ("weakness CWE-426", "cwe-426"),
        ("APT28 (G0007)", "g0007"),
        ("KONNI (S0356)", "s0356"),
        ("tactic TA0001", "ta0001"),
    ],
)
def test_identifiers_survive_as_single_tokens(text: str, token: str) -> None:
    assert token in tokenize(text)


def test_sub_technique_is_not_split_into_its_parent_and_a_number() -> None:
    """Split, T1055.001 would match every T1055 sub-technique and every '001'."""
    tokens = tokenize("T1055.001")

    assert tokens == ["t1055.001"]


def test_identifier_patterns_do_not_fire_inside_other_words() -> None:
    assert tokenize("xt1055 T10555") == ["xt1055", "t10555"]


def test_ordinary_words_split_on_punctuation_and_lowercase() -> None:
    assert tokenize("PowerShell.exe, Command-and-Control") == [
        "powershell",
        "exe",
        "command",
        "control",
    ]


def test_question_phrasing_is_dropped() -> None:
    assert tokenize("Which techniques does APT28 use?") == ["techniques", "apt28", "use"]


def test_term_ids_are_stable_across_processes() -> None:
    """Python's hash() is salted per process; an index built with it would be
    unqueryable from the next run, with every lexical score silently zero."""
    code = "from threatrag.index.sparse.bm25 import term_id; print(term_id('t1055.001'))"
    other = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()

    assert int(other) == term_id("t1055.001")
    assert 0 <= term_id("t1055.001") < 2**32


def test_document_weights_follow_bm25_term_frequency() -> None:
    encoder = Bm25Encoder(k1=1.2, b=0.75, avg_len=4.0)

    # 4 tokens, exactly avg_len, so length normalisation is k1.
    vector = encoder.encode_document("T1055 T1055 injection process")
    weights = dict(zip(vector.indices, vector.values, strict=True))

    assert weights[term_id("t1055")] == pytest.approx(2 * 2.2 / (2 + 1.2))
    assert weights[term_id("injection")] == pytest.approx(1 * 2.2 / (1 + 1.2))


def test_longer_documents_weigh_a_term_less() -> None:
    encoder = Bm25Encoder(avg_len=10.0)
    short = encoder.encode_document("T1055 injection")
    long = encoder.encode_document("T1055 " + "filler " * 30)

    def weight(vector: object) -> float:
        indices, values = vector.indices, vector.values  # type: ignore[attr-defined]
        return float(dict(zip(indices, values, strict=True))[term_id("t1055")])

    assert weight(long) < weight(short)


def test_query_weights_each_distinct_term_once() -> None:
    """The query side of BM25 is a sum of IDF over matched terms; Qdrant applies
    the IDF, so a repeated query word must not count twice."""
    vector = Bm25Encoder().encode_query("T1055 T1055 injection")

    assert sorted(vector.values) == [1.0, 1.0]
    assert vector.indices == sorted(vector.indices)


def test_a_question_of_only_stopwords_encodes_empty() -> None:
    vector = Bm25Encoder().encode_query("what is the")

    assert vector.indices == []
    assert vector.values == []
