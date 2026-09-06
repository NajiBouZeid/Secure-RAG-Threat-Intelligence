"""Inversion scoring.

The metrics have to disagree with each other in the right places, so most of
what is pinned here is the gap between reconstruction quality and disclosure: a
paraphrase that leaks nothing must not score as a breach, and a mangled string
that still names the secret must.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from threatrag.domain.types import Matrix, Vector
from threatrag.security.inversion.score import (
    corpus_bleu,
    score_reconstructions,
    token_f1,
    tokenize,
)


def _truth(
    chunk_id: str,
    text: str,
    *,
    secret_terms: Sequence[str] = (),
    source_type: str = "internal_note",
    tlp: str = "red",
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "secret_terms": list(secret_terms),
        "source_type": source_type,
        "tlp": tlp,
    }


class EchoEmbedder:
    """Embeds by token count, so a re-embedding differs only when the text does."""

    name = "stub"
    dim = 2

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return np.array([[float(len(tokenize(text))), 1.0] for text in texts], dtype=np.float32)

    def embed_query(self, text: str) -> Vector:
        return self.embed_documents([text])[0]


def test_tokenize_drops_punctuation_and_case() -> None:
    assert tokenize("TLP:RED -- CVE-2021-44228!") == ["tlp", "red", "cve", "2021", "44228"]


def test_token_f1_is_one_for_an_exact_reconstruction() -> None:
    assert token_f1(["a", "b"], ["a", "b"]) == 1.0


def test_token_f1_is_zero_when_nothing_overlaps() -> None:
    assert token_f1(["a", "b"], ["c", "d"]) == 0.0


def test_token_f1_counts_a_repeat_only_as_often_as_both_sides_have_it() -> None:
    """Padding the output with one repeated token must not manufacture recall."""
    padded = token_f1(["a", "b"], ["a", "a", "a", "a"])

    assert padded < token_f1(["a", "b"], ["a", "b"])


def test_corpus_bleu_is_one_for_identical_text() -> None:
    tokens = tokenize("the quick brown fox jumps over the lazy dog")

    assert corpus_bleu([tokens], [tokens]) == 1.0


def test_corpus_bleu_is_near_zero_when_no_ngram_survives() -> None:
    reference = tokenize("the quick brown fox jumps over the lazy dog")
    candidate = tokenize("entirely different wording with no shared phrases at all")

    assert corpus_bleu([reference], [candidate]) < 0.01


def test_corpus_bleu_penalises_a_short_reconstruction() -> None:
    reference = tokenize("the quick brown fox jumps over the lazy dog today")
    full = corpus_bleu([reference], [reference])
    truncated = corpus_bleu([reference], [reference[:5]])

    assert truncated < full


def test_a_paraphrase_that_leaks_nothing_is_not_counted_as_a_breach() -> None:
    truth = {"c1": _truth("c1", "APT29 used T1055 against the treasury", secret_terms=["APT29"])}

    scores = score_reconstructions(truth, {"c1": "a group employed process injection somewhere"})

    assert scores.overall.secret_recovery_rate == 0.0
    assert scores.rows[0].leaked is False


def test_a_mangled_reconstruction_that_names_the_secret_is_counted_as_one() -> None:
    """This is the case the text metrics miss and the defender cares about."""
    truth = {
        "c1": _truth(
            "c1",
            "The insider review found 41 repositories cloned by APT29",
            secret_terms=["41 repositories", "APT29"],
        )
    }

    scores = score_reconstructions(truth, {"c1": "review 41 repositories APT29 the the"})

    assert scores.rows[0].token_f1 < 0.8
    assert scores.rows[0].leaked is True
    assert scores.overall.secret_recovery_rate == 1.0


def test_partial_secret_recovery_is_not_a_full_leak() -> None:
    truth = {"c1": _truth("c1", "body", secret_terms=["APT29", "41 repositories"])}

    scores = score_reconstructions(truth, {"c1": "something about APT29"})

    assert scores.rows[0].secrets_recovered == 1
    assert scores.rows[0].leaked is False
    assert scores.overall.secret_recovery_rate == 0.5


def test_secret_matching_ignores_case() -> None:
    truth = {"c1": _truth("c1", "body", secret_terms=["Payment Approval Matrix"])}

    scores = score_reconstructions(truth, {"c1": "the payment approval matrix is attached"})

    assert scores.rows[0].leaked is True


def test_exact_match_survives_punctuation_and_case_differences() -> None:
    truth = {"c1": _truth("c1", "APT29 used T1055.")}

    scores = score_reconstructions(truth, {"c1": "apt29 used t1055"})

    assert scores.rows[0].exact_match is True


def test_round_trip_cosine_is_scored_only_when_an_embedder_is_supplied() -> None:
    truth = {"c1": _truth("c1", "one two three")}
    vectors = {"c1": np.array([3.0, 1.0], dtype=np.float32)}

    without = score_reconstructions(truth, {"c1": "one two three"})
    with_embedder = score_reconstructions(
        truth, {"c1": "one two three"}, stolen_vectors=vectors, embedder=EchoEmbedder()
    )

    assert without.rows[0].cosine is None
    assert with_embedder.rows[0].cosine is not None
    assert with_embedder.rows[0].cosine > 0.99


def test_results_are_broken_out_by_corpus_and_classification() -> None:
    truth = {
        "c1": _truth("c1", "note body", source_type="internal_note", tlp="red"),
        "c2": _truth("c2", "attack body", source_type="attack_cti", tlp="clear"),
    }

    scores = score_reconstructions(truth, {"c1": "note body", "c2": "unrelated"})

    by_type = {group.name: group for group in scores.by_source_type}
    by_tlp = {group.name: group for group in scores.by_tlp}
    assert by_type["internal_note"].exact_match_rate == 1.0
    assert by_type["attack_cti"].exact_match_rate == 0.0
    assert by_tlp["red"].count == 1


def test_reconstructions_with_no_answer_key_entry_are_reported() -> None:
    """A bundle brought back from a GPU run can carry ids the sample never had."""
    scores = score_reconstructions({"c1": _truth("c1", "body")}, {"c1": "body", "ghost": "text"})

    assert scores.scored == 1
    assert scores.unmatched == ["ghost"]


def test_targets_that_were_never_reconstructed_are_left_out_of_the_denominator() -> None:
    truth = {"c1": _truth("c1", "body"), "c2": _truth("c2", "other")}

    scores = score_reconstructions(truth, {"c1": "body"})

    assert scores.scored == 1
    assert scores.overall.count == 1
