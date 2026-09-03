# M3: CVE and vendor corpora, and what they cost retrieval

Run 2026-09-04. M3 roughly doubles the index and, more importantly, is the
first milestone where the corpus contains content that is *not* AUTHORITATIVE.
That matters more than "two more sources" suggests: M4's indirect injection
attack is a trust-tier failure, and it cannot be staged honestly against an
index where every document carries the same tier.

## What was added

| corpus | documents | chunks | share of index | TLP | trust tier |
|---|---|---|---|---|---|
| ATT&CK (M1) | 1757 | 20751 | 50.8% | CLEAR | AUTHORITATIVE |
| NVD CVE | 5170 | 17301 | 42.3% | CLEAR | AUTHORITATIVE |
| Vendor reports | 10 | 2729 | 6.7% | CLEAR | **VENDOR** |
| Internal notes (M2) | 16 | 34 | 0.1% | GREEN/AMBER/RED | VENDOR/COMMUNITY |
| **total** | **6953** | **40815** | | | |

The index went from 20785 to 40815 chunks: **it very nearly doubled.**

### CVE selection

Two passes, both reproducible:

* **170 CVEs ATT&CK discusses.** 171 were found in the bundle; one is cited by
  ATT&CK but has no NVD record, which the fetcher treats as data rather than as
  a failure.
* **5000 recent high-severity CVEs**, CRITICAL and HIGH, walked backwards in
  120-day windows from a **fixed** `window_end: 2026-09-01`. Fixed rather than
  `now()` deliberately: a window anchored to the current date would change the
  corpus between runs and quietly destroy comparability with M1 and M2.

Fetched in 176 requests over about two minutes with a registered API key. The
cache is per-request, so raising `recent_limit` later costs only the delta and
re-indexing costs nothing.

### Vendor reports

Ten reports, 73 MB, 1.15M characters of extracted text. Chosen for ATT&CK
technique density rather than brand: Red Canary's report is organised *by*
ATT&CK technique; Sophos, Talos and Unit 42 carry named tooling from incident
data; Mandiant and CrowdStrike name the same APT groups the gold set covers.
Dragos is a deliberate outlier -- ICS/OT is outside ATT&CK Enterprise.

Seven are publisher-hosted, three mirrored where the publisher gates the PDF.
The manifest is committed and the PDFs are not: they are copyrighted
publications, so the repo reproduces the corpus without redistributing it.

## Retrieval, same 200 questions

No new gold questions were added. Keeping the M1 set unchanged is the only
thing that makes these numbers comparable at all.

| metric | M1 (recursive, ATT&CK only) | M3 (all corpora) | change |
|---|---|---|---|
| recall@5 | 0.1641 | 0.1631 | −0.6% |
| recall@10 | 0.2629 | 0.2546 | −3.2% |
| MRR@10 | 0.3430 | 0.3372 | −1.7% |
| nDCG@10 | 0.2288 | 0.2236 | −2.3% |
| precision@10 | 0.1568 | 0.1530 | −2.4% |
| hit@10 | 0.660 | 0.640 | −3.0% |

Full M3 figures: at k=5, recall 0.1631, precision 0.1764, MRR 0.3133, nDCG
0.1824, hit 0.490. At k=10, recall 0.2546, precision 0.1530, MRR 0.3372, nDCG
0.2236, hit 0.640.

These are the numbers after the PDF cleaning fixes described below. Before them
the same evaluation gave recall@5 0.1614 and recall@10 0.2539, so removing
extraction junk from 6.7% of the index moved recall@5 by 0.0017. Small, in the
direction cleaner text should move it, and well inside what 200 queries can
resolve -- recorded because a fix that changed nothing measurable is worth
knowing about too.

### Reading this

**Doubling the index cost between 0.6% and 3.2% relative.** That is far less
than expected. 20,000 new chunks now compete for the same five slots, and the
gold answers were displaced from about one query in twenty-six at k=10.

The honest reading is that this measures *distractor robustness*, and the
result is mildly good news: MiniLM's ranking of ATT&CK content is not easily
displaced by topically adjacent CVE and vendor text. It is not evidence that
the retriever improved -- nothing about ATT&CK retrieval changed -- only that
it degraded gracefully.

The decline is real but small enough to sit near the edge of what 200 queries
can resolve. It should be read as "the corpus roughly doubled and retrieval
held", not as a precise measurement of harm.

**The corpus was not tuned to protect this number.** The cap stayed at the
agreed 5000 after the projection showed CVEs would be 42% of the index.
Shrinking it to get a nicer delta would have measured the experiment rather
than the system.

## The measurement that changed the corpus

Projecting chunk counts *before* ingesting caught a real defect. The first
shaping indexed each CVE's reference URLs, which turned out to be roughly 70%
of a CVE document. The recursive chunker split them into whole chunks of
nothing but links: one sampled CVE produced six chunks, three of them pure URL
lists and two of them 32-character orphans. Projected over the corpus that was
**26,005 chunks -- more than all of ATT&CK** -- for content that cannot answer
a question.

That failure would have been invisible in the results. The index grows, every
metric moves, and the conclusion "adding CVEs hurt retrieval" would have been
recorded when the truth was "adding forty thousand URLs hurt retrieval".

The section now carries what the references *mean* -- `Patch, Vendor Advisory,
VDB Entry (33 references)` -- which is the retrievable part. 16983 chunks
projected, 17301 actual, and the orphans are gone. Dropping the URLs is a
choice about what to index rather than a modification of NVD's content: the
canonical NVD URL on every document still reaches the complete reference list.

## Qualitative probes

Five queries against the live index, k=5:

| query shape | result |
|---|---|
| `What is CVE-2021-44228?` | correct CVE ranked 1st, but see below |
| Log4Shell described in prose | CVE-2021-44228, -45046, -44832: the right family |
| "most common initial access techniques in IR" | 4 of 5 are vendor reports (Mandiant, Unit 42, Dragos) |
| "threats targeting ICS and OT" | Dragos takes 3 of 5 |
| "techniques APT29 uses for persistence" | still all ATT&CK; not displaced |

The new corpora are retrievable and answer the question shapes they should.
ATT&CK-shaped questions still retrieve ATT&CK.

### A prediction that was half wrong, and a new problem

Before M3 I predicted that CVE identifiers would retrieve badly, because dense
MiniLM cannot match identifiers -- the weakness seen three times on ATT&CK ids.
The correct CVE *did* rank first, so the prediction was wrong as stated.

But the scores say something worse. The top five for `What is CVE-2021-44228?`
scored 0.826, 0.812, 0.812, 0.809, 0.808 -- and results two through five are
unrelated CVEs from 2026. The margin is 0.014. The model is not matching the
identifier; it is matching the *shape* of a CVE document, and the correct
answer leads by an amount indistinguishable from noise.

**This is partly self-inflicted, and it is new.** Every CVE document is built
from the same template with the same section headings, so 17,301 chunks are
near-identical in structure and their embeddings cluster tightly. Uniform
shaping made the corpus easier to read and harder to discriminate. The prose
query, which has real vocabulary to match, separates cleanly (0.740 → 0.589).

Worth carrying to M6 alongside BM25: lexical matching fixes identifier lookup
outright, and score compression across templated documents is an argument for
it that did not exist before M3.

## Carried forward

* **Multi-column PDFs still interleave.** Talos's report is laid out in
  columns and `pdfplumber` reads across them, so sentences from adjacent
  columns are spliced together mid-clause. Three cheaper artifacts were fixed
  once a spot check of the extracted text exposed them: headings arrived
  character-doubled (`TTHHRREEAATT`) because faux-bold is drawn twice, contents
  pages survived as dot leaders, and running feet whose page number changes
  defeated exact-match furniture detection. Column interleaving needs real
  layout analysis and is not fixed. It degrades a minority of one report rather
  than corrupting a corpus, and the extraction floor still catches the failure
  that matters -- a PDF with no recoverable text at all.
* **Chunk-level duplicates crowd top-k.** `T1053.005` occupied four of five
  slots in one probe. Scoring collapses chunks to their document so the metrics
  are unaffected, but generation wastes its context window on repeats. A
  per-document cap in the retriever would fix it; it changes retrieval, so it
  belongs with the M6 work, not here.
* **Tiny orphan chunks** come from the chunker's overlap carry and affect
  ATT&CK too. Not fixed: changing the chunker now would invalidate the M1
  baseline that these numbers are measured against.
* **The Sophos 2025 report cannot be ingested.** It is entirely rasterised --
  70 to 100 images per page, zero extractable characters across 25 pages. The
  extraction floor rejected it, which is exactly what that floor exists for; an
  empty document would otherwise have entered the index under a trusted
  vendor's name and silently retrieved nothing forever. OCR was rejected as a
  fix: it trades a loud failure for transcription errors in precisely the
  malware names and technical terms retrieval depends on. The 2026 edition is
  used instead.
* **NVD terms of use** are honoured in code rather than by intention: pacing at
  the published limit with no concurrency, bounded retries that surrender
  rather than hammer, the key in a header so it cannot leak through a logged
  URL, and the required attribution notice displayed by the application. The
  obligations are asserted by tests, because the likeliest future edit to that
  module is someone trying to make the fetch faster.
* **M4 constraint.** Poisoning and injection must operate on synthetic
  `SYNTHETIC_ADVERSARIAL` documents, never on tampered NVD or vendor records
  still labelled as theirs. That is both what the NVD terms require and better
  experimental hygiene: an authored attack document has known ground truth, a
  corrupted real one does not.

This product uses the NVD API but is not endorsed or certified by the NVD.
