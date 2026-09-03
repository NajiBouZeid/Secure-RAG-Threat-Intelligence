# M1 baseline: undefended retrieval over ATT&CK

First end-to-end run against real data, 2026-09-02. This is the utility
baseline every later defence is scored against, so the conditions matter as
much as the numbers.

## Conditions

| | |
|---|---|
| Corpus | MITRE ATT&CK Enterprise, `mitre-attack/attack-stix-data` (53.8 MB, 26086 STIX objects) |
| Documents | 1757 (697 technique, 825 software, 176 group, 44 mitigation, 15 tactic) |
| Chunks | 19833, `structural` strategy, 512 chars / 64 overlap |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2`, 384-d, cosine, CPU |
| Store | Qdrant v1.19.0, named vector `all-minilm` |
| Gold set | 200 queries from MITRE `intrusion-set --uses--> attack-pattern` edges, seed 1337 |
| Defences | none (undefended baseline) |

Scoring collapses chunks to their document's ATT&CK id before comparing
against the gold labels, so several chunks of one technique count once rather
than inflating recall.

## Results

| metric | k=5 | k=10 |
|---|---|---|
| recall@k | 0.1392 | 0.2311 |
| precision@k | 0.1551 | 0.1413 |
| MRR | 0.2803 | 0.3112 |
| nDCG@k | 0.1538 | 0.1985 |
| hit_rate | 0.475 | 0.655 |

## Reading these numbers

They are modest, and honestly so. The gold questions are relational -- "which
credential access techniques does Ke3chang use?" -- and answering one requires
surfacing 3 to 8 specific techniques out of 697. A single dense vector over
MiniLM is weak at that: it matches the *topic* (credential access) far more
strongly than the *actor* (Ke3chang). Precision falling from k=5 to k=10 while
recall nearly doubles is the signature of exactly that -- the right documents
are present but ranked below topically similar noise.

That leaves real headroom, which is what a baseline is for. Hybrid sparse-dense
retrieval, a reranker, or query decomposition should each move it, and the
chunking comparison below is the first cheap experiment.

## What this run cost to establish

An earlier version of this table read `recall@5 = 0.034`, which is chance. The
retriever was not at fault: the corpus did not contain the evidence. Which
techniques a group uses is recorded only as a STIX `uses` edge, and only 5 of
941 gold pairs had that evidence in any indexed document. Folding ATT&CK's
procedure examples into the technique documents took coverage to 941/941 and
the metrics to the table above.

The lesson generalises past this project: before trusting any retrieval metric,
verify the answer is physically present in the indexed text. A retrieval score
cannot distinguish "the retriever is bad" from "the answer was never there".

## Chunking comparison

Run 2026-09-03. Every variable except the chunker is held fixed: same corpus,
same 200-query gold set (seed 1337), same MiniLM embedder, same scoring. Each
strategy indexes its own Qdrant collection, so all three coexist and can be
re-scored without a re-ingest.

| strategy | chunks | recall@5 | recall@10 | precision@10 | MRR@10 | nDCG@10 | hit@10 |
|---|---|---|---|---|---|---|---|
| structural | 19833 | 0.1392 | 0.2311 | 0.1413 | 0.3112 | 0.1985 | 0.655 |
| fixed | 14836 | 0.1415 | 0.2199 | 0.1376 | 0.3048 | 0.1907 | 0.620 |
| **recursive** | 20751 | **0.1641** | **0.2629** | **0.1568** | **0.3430** | **0.2288** | **0.660** |

Structural at k=10 reproduced the previous day's `0.2311` to four decimals,
which is the reproducibility check that makes the other two rows comparable.

**Recursive wins, and by enough to believe.** It is ahead of structural on
every metric at both cutoffs -- recall@5 +17.9% relative, recall@10 +13.8%,
MRR +10.2%. That is well clear of the noise floor.

**Fixed-width is a tie that becomes a loss.** It edges structural at k=5 by
margins worth nothing (hit_rate 0.485 vs 0.475 is two queries out of 200), then
falls behind at k=10 across the board. Read the pair together and the fair
summary is that naive fixed-width chunking is no better than structural and
somewhat worse deeper in the ranking, on 26% fewer chunks.

**Why recursive plausibly wins -- untested.** Structural splits on ATT&CK's own
section headings, which vary wildly in length; a long Procedure Examples
section becomes one chunk mixing dozens of groups. Fixed-width ignores
boundaries entirely. Recursive sits between: it descends a separator hierarchy,
so it keeps each procedure-example paragraph intact while still bounding chunk
size. Since the gold set asks exactly which group uses which technique, and
that evidence lives in the procedure examples (see the section above), keeping
one group's paragraph un-merged is the mechanism that should matter. This is a
hypothesis consistent with the chunk counts, not a measured claim -- confirming
it means inspecting which chunks changed rank, not just the aggregate.

**What it does not fix.** Recursive lifts recall@5 from 0.139 to 0.164; the
ceiling is still low, and precision still falls from k=5 to k=10. Chunking is
worth ~15% here, so the dominant error remains the one the baseline diagnosed:
a single dense vector matches tactic over actor. Hybrid sparse-dense retrieval,
a reranker, or query decomposition are the levers with real headroom, and the
defence-vs-utility curve in M7 will read better from a higher starting utility.
