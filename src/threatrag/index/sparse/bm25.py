"""BM25 term weights for Qdrant's native sparse vectors.

MiniLM places ``T1055.001`` and ``T1055.002`` next to each other, because to a
dense encoder they are the same string with one digit changed. An analyst
asking for one of them wants exactly that one, and a lexical match is the
retriever that can tell them apart. This module produces the lexical half.

The split of work with Qdrant is deliberate. BM25 is ``idf(term) * tf_part``;
the collection is created with ``Modifier.IDF`` so Qdrant computes the IDF
from the corpus it actually holds, and this module supplies only the
per-document term-frequency part. Computing IDF here would freeze it at build
time, and a poison document written later by the attack harness would then be
scored against statistics that do not include it.

Two choices are load-bearing:

* **Identifiers are single tokens.** A generic tokenizer splits ``T1055.001``
  into ``t1055`` and ``001`` and ``CVE-2024-3094`` into three numbers, which
  turns the one query a lexical retriever should win into a bag of digits that
  matches every sub-technique and every CVE from the same year.
* **Term ids are a stable hash**, not Python's ``hash()``, which is salted per
  process: a collection built in one run would be queried with different ids in
  the next, and every lexical score would silently be zero.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

# Tried in order, so an identifier is claimed before the generic word rule can
# split it. Identifiers must stand alone, so "xt1055" or "T10555" stay ordinary
# words. Case-insensitive, and lowercased afterwards.
_TOKEN = re.compile(
    r"""
    (?<![a-z0-9])
    (?:
        cve-\d{4}-\d{4,}        # CVE-2024-3094
        | cwe-\d+               # CWE-426
        | t\d{4}(?:\.\d{3})?    # T1055, T1055.001
        | (?:ta|[gsmc])\d{4}    # TA0001, G0007, S0356, M1038, C0024
    )
    (?![a-z0-9])
    | [a-z0-9]+                 # everything else, split on punctuation
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Only words that carry no meaning in any question this corpus is asked. IDF
# already discounts common terms; this list exists so that a question's
# phrasing ("which ... does ... use") does not contribute matches of its own.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "does",
        "do",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "which",
        "who",
        "with",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lowercased tokens with identifiers kept whole and stopwords removed."""
    return [
        token
        for token in (match.lower() for match in _TOKEN.findall(text))
        if token not in STOPWORDS
    ]


def term_id(token: str) -> int:
    """A process-independent 32-bit id; Qdrant sparse indices are uint32."""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "big")


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class Bm25Encoder:
    """BM25's term-frequency component, with IDF left to the index.

    ``avg_len`` is the mean token count of an indexed passage. It is a fixed
    parameter rather than a statistic recomputed on every write, so a document
    indexed later is weighted on the same scale as the corpus it joins.
    """

    k1: float = 1.2
    b: float = 0.75
    avg_len: float = 100.0

    def encode_document(self, text: str) -> SparseVector:
        tokens = tokenize(text)
        norm = self.k1 * (1 - self.b + self.b * len(tokens) / self.avg_len)
        weights: dict[int, float] = {}
        for token, tf in Counter(tokens).items():
            _add(weights, term_id(token), tf * (self.k1 + 1) / (tf + norm))
        return _sparse(weights)

    def encode_query(self, text: str) -> SparseVector:
        # Each distinct term once, weight 1: the query side of BM25 is the sum
        # of IDF over matched terms, and Qdrant applies the IDF.
        weights: dict[int, float] = {}
        for token in dict.fromkeys(tokenize(text)):
            _add(weights, term_id(token), 1.0)
        return _sparse(weights)


def _add(weights: dict[int, float], index: int, value: float) -> None:
    # Two tokens hashing to one id is possible at 2^32 and harmless if summed;
    # overwriting would drop a term instead.
    weights[index] = weights.get(index, 0.0) + value


def _sparse(weights: dict[int, float]) -> SparseVector:
    indices = sorted(weights)
    return SparseVector(indices=indices, values=[weights[index] for index in indices])
