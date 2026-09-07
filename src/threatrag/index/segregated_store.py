"""D5 -- keep confidential material out of the collection an attacker steals.

M5's two attacks never issue a query. They operate on stored vectors, so every
defence at the retrieval or answer boundary is downstream of the breach and
cannot touch them: the ACL governs answers, and neither attack asks a question.
What *does* touch them is topology. An attacker who steals the public index
finds 40781 public chunks and no confidential note to invert or re-identify,
because the notes were never in that collection.

This routes by classification, not by corpus: anything above the public
threshold is written to a second collection. A vendor report marked AMBER is
segregated for the same reason an internal note is, which is the property that
makes this a policy rather than a special case for one YAML file.

**Retrieval quality is unaffected, and that is a measured claim rather than a
hope.** With one embedder and one distance metric, scores from the two
collections are directly comparable, and any chunk in the global top-k is
necessarily in its own collection's top-k. Searching both and merging by score
therefore returns exactly the ranking a single collection would have. The cost
of this defence is operational -- two searches per query, two collections to
provision and back up, and a second store to secure -- not a worse answer. That
is a real finding for M7 and it is worth stating plainly instead of inventing a
retrieval penalty to make the trade-off look symmetrical.

An uncleared principal never causes the restricted collection to be queried at
all, so the segregation also narrows what an over-broad query can reach even
before anything is stolen.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

from threatrag.domain.models import TLP, Chunk, Principal, RetrievedChunk
from threatrag.domain.ports import VectorStore
from threatrag.domain.types import Matrix, Vector


class SegregatedStore:
    """Two collections behind one ``VectorStore``, split on TLP."""

    def __init__(
        self,
        public: VectorStore,
        restricted: VectorStore,
        *,
        restrict_above: TLP = TLP.GREEN,
    ) -> None:
        self._public = public
        self._restricted = restricted
        self._restrict_above = restrict_above

    @property
    def collection(self) -> str:
        return f"{self._public.collection}+{self._restricted.collection}"

    def _is_restricted(self, chunk: Chunk) -> bool:
        return chunk.tlp.rank > self._restrict_above.rank

    def ensure_collection(self, vectors: Mapping[str, int]) -> None:
        self._public.ensure_collection(vectors)
        self._restricted.ensure_collection(vectors)

    def upsert(self, vector_name: str, chunks: Sequence[Chunk], vectors: Matrix) -> int:
        written = 0
        for store, selected in self._split(chunks):
            if not selected:
                continue
            indices = [index for index, _ in selected]
            written += store.upsert(
                vector_name,
                [chunk for _, chunk in selected],
                vectors[indices],
            )
        return written

    def _split(self, chunks: Sequence[Chunk]) -> list[tuple[VectorStore, list[tuple[int, Chunk]]]]:
        public: list[tuple[int, Chunk]] = []
        restricted: list[tuple[int, Chunk]] = []
        for index, chunk in enumerate(chunks):
            (restricted if self._is_restricted(chunk) else public).append((index, chunk))
        return [(self._public, public), (self._restricted, restricted)]

    def attach_vectors(self, vector_name: str, vectors: Mapping[str, Vector]) -> int:
        # Chunk ids are content-addressed and a chunk lives in exactly one
        # collection, so offering the whole batch to both is safe: each store
        # skips the ids it does not hold.
        return self._public.attach_vectors(vector_name, vectors) + self._restricted.attach_vectors(
            vector_name, vectors
        )

    def scroll_chunks(
        self, *, source_type: str | None = None, batch_size: int = 256
    ) -> Iterator[Chunk]:
        yield from self._public.scroll_chunks(source_type=source_type, batch_size=batch_size)
        yield from self._restricted.scroll_chunks(source_type=source_type, batch_size=batch_size)

    def get_vectors(self, vector_name: str, chunk_ids: Sequence[str]) -> dict[str, Vector]:
        found = self._public.get_vectors(vector_name, chunk_ids)
        found.update(self._restricted.get_vectors(vector_name, chunk_ids))
        return found

    def search(
        self,
        vector_name: str,
        query_vector: Vector,
        k: int,
        principal: Principal | None = None,
    ) -> list[RetrievedChunk]:
        results = list(self._public.search(vector_name, query_vector, k, principal=principal))
        # Skipped entirely for a principal who could not read the results
        # anyway: the restricted collection is not touched by an uncleared
        # query, rather than queried and then filtered.
        if principal is None or principal.clearance.rank > self._restrict_above.rank:
            results += self._restricted.search(vector_name, query_vector, k, principal=principal)
        # Exact, not approximate: any chunk in the global top-k is in its own
        # collection's top-k, so merging two k-sized lists by score reproduces
        # the single-collection ranking.
        results.sort(key=lambda hit: hit.score, reverse=True)
        return results[:k]

    def count(self) -> int:
        return self._public.count() + self._restricted.count()

    def count_with_vector(self, vector_name: str) -> int:
        return self._public.count_with_vector(vector_name) + self._restricted.count_with_vector(
            vector_name
        )

    def delete_by_source_type(self, source_type: str) -> int:
        return self._public.delete_by_source_type(
            source_type
        ) + self._restricted.delete_by_source_type(source_type)
