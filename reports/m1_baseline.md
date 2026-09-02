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

## Not yet run

The chunking comparison. `configs/experiments/chunking_fixed.yaml` and
`chunking_recursive.yaml` each target their own collection, so all three
strategies can be evaluated against this same gold set without colliding.
Each needs a full re-ingest (about 7 minutes on CPU).
