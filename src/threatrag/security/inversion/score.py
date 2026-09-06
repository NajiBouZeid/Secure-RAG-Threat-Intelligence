"""Scoring reconstructions against the text they were inverted from.

Four numbers, answering different questions. Exact match and token F1 say how
close the text came. Corpus BLEU is here because it is what the vec2text
literature reports, so this result can be read against theirs. Round-trip
cosine -- the similarity between the stolen vector and the re-embedding of the
reconstruction -- says whether the attack converged in vector space even when
the surface text drifted, which is the difference between a failed attack and
a paraphrase.

The fifth is the one a defender should read first. Secret recovery asks whether
the terms that made a chunk confidential survived the round trip at all. A
reconstruction can score badly on every text metric and still name the actor,
the CVE and the finding, and a fluent paraphrase can score well and disclose
nothing. Reconstruction quality and disclosure are not the same question.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from threatrag.domain.ports import Embedder

_TOKEN = re.compile(r"[a-z0-9]+")
_MAX_NGRAM = 4


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric runs.

    Deliberately crude and shared by every text metric, so a difference between
    two numbers is a difference in the reconstruction rather than in how each
    metric happened to split the string.
    """
    return _TOKEN.findall(text.lower())


class TargetScore(BaseModel):
    chunk_id: str
    source_type: str = ""
    tlp: str = ""
    exact_match: bool = False
    token_f1: float = 0.0
    cosine: float | None = None
    secret_terms: int = 0
    secrets_recovered: int = 0

    @property
    def leaked(self) -> bool:
        """Every term that made this chunk confidential survived the round trip."""
        return self.secret_terms > 0 and self.secrets_recovered == self.secret_terms


class GroupScore(BaseModel):
    name: str
    count: int = 0
    exact_match_rate: float = 0.0
    mean_token_f1: float = 0.0
    mean_cosine: float | None = None
    secret_recovery_rate: float = 0.0
    fully_leaked: int = 0


class InversionScores(BaseModel):
    rows: list[TargetScore] = Field(default_factory=list)
    corpus_bleu: float = 0.0
    overall: GroupScore = GroupScore(name="overall")
    by_source_type: list[GroupScore] = Field(default_factory=list)
    by_tlp: list[GroupScore] = Field(default_factory=list)
    scored: int = 0
    unmatched: list[str] = Field(default_factory=list)


def token_f1(reference: Sequence[str], candidate: Sequence[str]) -> float:
    """Bag-of-tokens F1, counting repeats only as often as both sides have them."""
    if not reference and not candidate:
        return 1.0
    if not reference or not candidate:
        return 0.0

    overlap = sum((Counter(reference) & Counter(candidate)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(candidate)
    recall = overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def corpus_bleu(references: Sequence[Sequence[str]], candidates: Sequence[Sequence[str]]) -> float:
    """BLEU-4 with a brevity penalty, aggregated over the corpus.

    Corpus-level rather than averaged per sentence: with chunk-length text a
    single missing 4-gram zeroes a sentence score, and the mean of those is
    dominated by which sentences happened to fail rather than by how much was
    recovered.
    """
    if not references:
        return 0.0

    clipped = [0] * _MAX_NGRAM
    totals = [0] * _MAX_NGRAM
    ref_len = 0
    cand_len = 0

    for reference, candidate in zip(references, candidates, strict=True):
        ref_len += len(reference)
        cand_len += len(candidate)
        for order in range(1, _MAX_NGRAM + 1):
            ref_counts = _ngrams(reference, order)
            cand_counts = _ngrams(candidate, order)
            totals[order - 1] += max(len(candidate) - order + 1, 0)
            clipped[order - 1] += sum((cand_counts & ref_counts).values())

    if cand_len == 0:
        return 0.0

    log_precision = 0.0
    for order in range(_MAX_NGRAM):
        if totals[order] == 0:
            return 0.0
        # A floor on the numerator where nothing matched, so a corpus with no
        # 4-gram overlap scores near zero instead of being undefined.
        numerator = float(clipped[order]) if clipped[order] > 0 else 1e-9
        log_precision += math.log(numerator / totals[order]) / _MAX_NGRAM

    brevity = 1.0 if cand_len > ref_len else math.exp(1 - ref_len / max(cand_len, 1))
    return brevity * math.exp(log_precision)


def _ngrams(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1))


def _cosine(left: np.ndarray[Any, Any], right: np.ndarray[Any, Any]) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _recovered(terms: Sequence[str], reconstruction: str) -> int:
    """Substring match on lowercased text.

    Generous on purpose: the defender's question is whether the term is legible
    in the output, not whether the model reproduced the surrounding wording.
    """
    haystack = reconstruction.lower()
    return sum(1 for term in terms if term and term.lower() in haystack)


def _group(name: str, rows: Sequence[TargetScore]) -> GroupScore:
    if not rows:
        return GroupScore(name=name)

    cosines = [row.cosine for row in rows if row.cosine is not None]
    with_secrets = [row for row in rows if row.secret_terms > 0]
    recovered = sum(row.secrets_recovered for row in with_secrets)
    total_terms = sum(row.secret_terms for row in with_secrets)
    return GroupScore(
        name=name,
        count=len(rows),
        exact_match_rate=sum(row.exact_match for row in rows) / len(rows),
        mean_token_f1=sum(row.token_f1 for row in rows) / len(rows),
        mean_cosine=(sum(cosines) / len(cosines)) if cosines else None,
        secret_recovery_rate=(recovered / total_terms) if total_terms else 0.0,
        fully_leaked=sum(1 for row in rows if row.leaked),
    )


def score_reconstructions(
    truth: Mapping[str, Mapping[str, Any]],
    reconstructions: Mapping[str, str],
    *,
    stolen_vectors: Mapping[str, np.ndarray[Any, Any]] | None = None,
    embedder: Embedder | None = None,
) -> InversionScores:
    """Score every reconstruction that has a matching answer-key entry.

    Round-trip cosine is computed only when both an embedder and the stolen
    vectors are supplied, so scoring works offline on a bundle brought back
    from a GPU run and gains the extra column when the encoder is available.
    """
    rows: list[TargetScore] = []
    references: list[list[str]] = []
    candidates: list[list[str]] = []
    unmatched = sorted(set(reconstructions) - set(truth))

    round_trip: dict[str, np.ndarray[Any, Any]] = {}
    if embedder is not None and stolen_vectors:
        scorable = [cid for cid in truth if cid in reconstructions and cid in stolen_vectors]
        if scorable:
            matrix = embedder.embed_documents([reconstructions[cid] for cid in scorable])
            round_trip = dict(zip(scorable, matrix, strict=True))

    for chunk_id, record in truth.items():
        if chunk_id not in reconstructions:
            continue

        reference = tokenize(str(record.get("text", "")))
        candidate = tokenize(reconstructions[chunk_id])
        references.append(reference)
        candidates.append(candidate)

        terms = [str(term) for term in record.get("secret_terms", [])]
        cosine = None
        if chunk_id in round_trip and stolen_vectors is not None:
            cosine = _cosine(stolen_vectors[chunk_id], round_trip[chunk_id])

        rows.append(
            TargetScore(
                chunk_id=chunk_id,
                source_type=str(record.get("source_type", "")),
                tlp=str(record.get("tlp", "")),
                exact_match=reference == candidate,
                token_f1=token_f1(reference, candidate),
                cosine=cosine,
                secret_terms=len(terms),
                secrets_recovered=_recovered(terms, reconstructions[chunk_id]),
            )
        )

    return InversionScores(
        rows=rows,
        corpus_bleu=corpus_bleu(references, candidates),
        overall=_group("overall", rows),
        by_source_type=[
            _group(name, [row for row in rows if row.source_type == name])
            for name in sorted({row.source_type for row in rows})
        ],
        by_tlp=[
            _group(name, [row for row in rows if row.tlp == name])
            for name in sorted({row.tlp for row in rows})
        ],
        scored=len(rows),
        unmatched=unmatched,
    )
