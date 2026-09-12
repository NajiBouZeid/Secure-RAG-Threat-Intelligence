"""Answer-level utility scoring.

The load-bearing behaviour is that a refusal is a miss. A defence that declines
to answer has protected the system by not being one, and if refusals were merely
excluded from the denominator that cost would vanish from the plot.
"""

from __future__ import annotations

import pytest

from threatrag.domain.models import Answer, Chunk, RetrievedChunk, SourceType
from threatrag.eval.answers import score_answers, technique_ids, ungrounded_ids
from threatrag.eval.goldset import GoldQuery

NO_EVIDENCE = "No indexed material matched this question, so there is nothing to answer from."


def _query(*relevant: str) -> GoldQuery:
    return GoldQuery(id="q", question="Which techniques?", relevant=list(relevant))


def _answer(text: str, **kwargs: object) -> Answer:
    return Answer(question="Which techniques?", text=text, **kwargs)  # type: ignore[arg-type]


def test_an_answer_naming_a_relevant_technique_scores() -> None:
    scores = score_answers(
        [_answer("APT29 uses T1071.001 for command and control [1].")],
        [_query("T1071.001", "T1105")],
        no_evidence=NO_EVIDENCE,
    )

    assert scores.answer_utility == 1.0


def test_the_parent_id_alone_does_not_satisfy_a_sub_technique() -> None:
    """Matching the parent would inflate the score, which is the direction that
    matters."""
    scores = score_answers(
        [_answer("APT29 uses T1071 for command and control.")],
        [_query("T1071.001")],
        no_evidence=NO_EVIDENCE,
    )

    assert scores.answer_utility == 0.0


def test_a_blocked_answer_counts_as_a_refusal_and_a_miss() -> None:
    """D6 buys its attack numbers by refusing; the cost has to land on utility
    in the same table."""
    blocked = _answer("Refusing.", blocked=True)

    scores = score_answers([blocked], [_query("T1071.001")], no_evidence=NO_EVIDENCE)

    assert scores.refusal_rate == 1.0
    assert scores.answer_utility == 0.0


def test_the_no_evidence_path_is_a_refusal_too() -> None:
    scores = score_answers([_answer(NO_EVIDENCE)], [_query("T1071.001")], no_evidence=NO_EVIDENCE)

    assert scores.refusal_rate == 1.0


def test_a_refusal_that_happens_to_contain_the_id_still_does_not_score() -> None:
    """D6's refusal text names the subject it refused about. Counting that as
    utility would make the defence look free."""
    blocked = _answer("Refusing an uncorroborated claim about T1071.001.", blocked=True)

    scores = score_answers([blocked], [_query("T1071.001")], no_evidence=NO_EVIDENCE)

    assert scores.answer_utility == 0.0
    assert scores.refusal_rate == 1.0


def test_unsupported_citations_are_counted_separately_from_utility() -> None:
    """An answer can be on target and still have invented a source."""
    answer = _answer("APT29 uses T1071.001 [9].", unsupported_citations=["[9]"])

    scores = score_answers([answer], [_query("T1071.001")], no_evidence=NO_EVIDENCE)

    assert scores.answer_utility == 1.0
    assert scores.unsupported_citation_rate == 1.0


def test_mismatched_lengths_are_refused() -> None:
    """Silently zipping to the shorter list would score a truncated run as a
    complete one."""
    with pytest.raises(ValueError, match="1 answers but 2 queries"):
        score_answers([_answer("x")], [_query("T1"), _query("T2")], no_evidence=NO_EVIDENCE)


def test_an_empty_run_does_not_divide_by_zero() -> None:
    scores = score_answers([], [], no_evidence=NO_EVIDENCE)

    assert scores.as_row()["answer_utility"] == 0.0
    assert scores.as_row()["answer_recall"] == 0.0
    assert scores.as_row()["ungrounded_id_rate"] == 0.0


def _passage(text: str, ref: str = "T1071.001") -> RetrievedChunk:
    chunk = Chunk(
        id=f"{ref}#0",
        doc_id=ref,
        ordinal=0,
        text=text,
        title=f"{ref} Application Layer Protocol (technique)",
        source_type=SourceType.ATTACK_CTI,
        source_ref=ref,
    )
    return RetrievedChunk(chunk=chunk, score=0.5)


def test_recall_is_graded_by_how_many_relevant_techniques_are_named() -> None:
    """The binary score cannot tell one of four from four of four."""
    scores = score_answers(
        [_answer("Uses T1071.001 and T1105.")],
        [_query("T1071.001", "T1105", "T1027", "T1059")],
        no_evidence=NO_EVIDENCE,
    )

    assert scores.answer_utility == 1.0
    assert scores.answer_recall == 0.5


def test_a_refusal_scores_zero_recall_and_still_counts_in_the_mean() -> None:
    scores = score_answers(
        [_answer("Uses T1071.001.", retrieved=[_passage("x")]), _answer("No.", blocked=True)],
        [_query("T1071.001"), _query("T1071.001")],
        no_evidence=NO_EVIDENCE,
    )

    assert scores.answer_recall == 0.5


def test_an_id_found_in_no_retrieved_passage_is_ungrounded() -> None:
    answer = _answer(
        "Uses T1071.001 and T1486.",
        retrieved=[_passage("C2 over HTTPS, see T1071.001.")],
    )

    assert ungrounded_ids(answer) == ["T1486"]


def test_the_parent_of_a_retrieved_sub_technique_is_grounded() -> None:
    """Generalising from the evidence is not inventing it."""
    answer = _answer("Uses T1071.", retrieved=[_passage("C2 over HTTPS.")])

    assert ungrounded_ids(answer) == []


def test_a_sub_technique_of_a_retrieved_parent_is_not_grounded() -> None:
    """The opposite direction adds specificity the evidence never gave."""
    answer = _answer("Uses T1059.001.", retrieved=[_passage("Scripting.", ref="T1059")])

    assert ungrounded_ids(answer) == ["T1059.001"]


def test_the_ungrounded_rate_is_over_ids_named_not_over_questions() -> None:
    answer = _answer(
        "T1071.001, T1105 and T1486.",
        retrieved=[_passage("T1105 appears here too.")],
    )

    scores = score_answers([answer], [_query("T1071.001")], no_evidence=NO_EVIDENCE)

    assert scores.named_ids == 3
    assert scores.ungrounded == 1


def test_every_question_keeps_a_record_in_order() -> None:
    """Without per-question records two cells cannot be compared question by
    question, and a later metric would need a new run."""
    scores = score_answers(
        [_answer("Uses T1071.001.", retrieved=[_passage("x")]), _answer("No.", blocked=True)],
        [
            GoldQuery(id="a", question="?", relevant=["T1071.001"]),
            GoldQuery(id="b", question="?", relevant=["T1105"]),
        ],
        no_evidence=NO_EVIDENCE,
    )

    assert [record.id for record in scores.records] == ["a", "b"]
    assert scores.records[0].recall == 1.0
    assert scores.records[1].refused is True
    assert scores.records[1].as_json()["text"] == "No."


def test_technique_ids_are_deduplicated_in_first_seen_order() -> None:
    assert technique_ids("T1105 then [T1071.001], T1105 again") == ["T1105", "T1071.001"]
