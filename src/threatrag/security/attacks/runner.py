"""Running an attack against the live pipeline, then cleaning up after it.

The runner is the harness M7 will later call once per defence configuration. For
now it does the honest minimum: index the poisoned document into the same
collection the real corpus lives in, ask the target query through the ordinary
answer pipeline under the attack's principal, observe what happened, and
*always* remove the poison again.

Two things are deliberate.

*The poison is additive and exactly reversible.* It is indexed with its own
``SYNTHETIC_ADVERSARIAL`` source type and deleted by that type in a ``finally``
block, so a run leaves the collection byte-for-byte as it found it and the M1/M3
baseline numbers are never disturbed -- even if a run raises. It also means the
poison competes against the real corpus, which is what makes displacement in a
poisoning attack a real measurement rather than a staged one.

*The render step is only simulated for exfiltration.* Firing the beacon models
what M2's owned renderer does to a Markdown image; doing it for every family
would be measuring nothing for the other two.
"""

from __future__ import annotations

from pydantic import BaseModel

from threatrag.domain.models import Principal, SourceType
from threatrag.domain.ports import VectorStore
from threatrag.ingest.pipeline import IngestPipeline
from threatrag.rag.pipeline import AnswerPipeline
from threatrag.security.attacks.evaluate import CriterionResult, evaluate
from threatrag.security.attacks.loader import attack_to_document
from threatrag.security.attacks.render import render_and_fire
from threatrag.security.attacks.schema import Attack, AttackFamily
from threatrag.security.attacks.sink import ExfiltrationSink


class AttackResult(BaseModel):
    attack_id: str
    family: AttackFamily
    target_query: str
    succeeded: bool
    criteria: list[CriterionResult]
    answer_text: str
    retrieved_refs: list[str]
    fired_urls: list[str] = []


class AttackRunner:
    """Index a poison document, run one query, evaluate, and clean up."""

    def __init__(
        self,
        pipeline: AnswerPipeline,
        ingest: IngestPipeline,
        store: VectorStore,
        *,
        sink: ExfiltrationSink | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._ingest = ingest
        self._store = store
        self._sink = sink

    def run(self, attack: Attack) -> AttackResult:
        sink_base = self._sink.base_url if self._sink is not None else None
        document = attack_to_document(attack, sink_base=sink_base)
        if self._sink is not None:
            self._sink.clear()

        try:
            self._ingest.run([document])
            principal = Principal(id=f"attack:{attack.id}", clearance=attack.clearance)
            answer = self._pipeline.answer(attack.target_query, principal=principal)

            fired: list[str] = []
            if attack.family is AttackFamily.EXFILTRATION and self._sink is not None:
                fired = render_and_fire(answer.text, self._sink.base_url)

            received = list(self._sink.received) if self._sink is not None else []
            criteria = evaluate(attack, answer=answer, sink_received=received)
            return AttackResult(
                attack_id=attack.id,
                family=attack.family,
                target_query=attack.target_query,
                succeeded=all(result.passed for result in criteria),
                criteria=criteria,
                answer_text=answer.text,
                retrieved_refs=[hit.chunk.source_ref for hit in answer.retrieved],
                fired_urls=fired,
            )
        finally:
            # Guaranteed removal. The poison never outlives its own run, so the
            # collection the real corpus shares is restored even on failure.
            self._store.delete_by_source_type(SourceType.SYNTHETIC_ADVERSARIAL.value)
