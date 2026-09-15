# M5: what a stolen vector index gives back

Run 2026-09-06 against the M3 index (40815 chunks). M4 attacked the system
through its inputs — poisoned documents that the pipeline retrieves and the
model reads. M5 attacks it through its storage instead, and assumes a different
breach: the attacker holds the vectors. That is not a hypothetical shape for
this project, which runs Qdrant unauthenticated on `localhost:6333`; a stolen
snapshot or an exposed port is the same starting position.

The question is what vectors alone give up. Two attacks answer it differently,
and the split is forced by which encoders have public tooling.

## The sample, and why it is 334 chunks and not 40815

Inverting a vector is independent of every other vector, so a reconstruction
rate measured on a fair sample is the rate for the index. Embedding all 40815
chunks with a second encoder would spend an hour of CPU to reproduce the same
percentage. What the sample has to earn instead is fairness, so it is drawn
under two rules, seeded (20260906) and recorded with every result:

- **The internal notes are taken whole** — all 34. They are the only AMBER and
  RED chunks in the corpus, they are the documents whose leakage is the point
  of the exercise, and at 34 of 40815 a proportional draw would have taken one.
- **Everything else is proportional** to how much of the index it occupies,
  allocated by largest remainder: `attack_cti` 153, `nvd_cve` 127,
  `vendor_report` 20.

Attacker-authored documents are excluded, so leftover M4 poison cannot inflate
a reconstruction rate with text the attacker wrote himself.

## Getting a second encoder's vectors in without destroying the first

The system retrieves with `all-MiniLM-L6-v2`; the inversion attack needs
`gtr-t5-base`, the only strong encoder with a public corrector. The collection
has declared both named vectors since M1, and the store's own docstring said
points "may carry any subset of them, which is what lets the MiniLM ingest and
a later GTR ingest share one chunk set."

That was not true of the code. Qdrant replaces a point on upsert, so the
documented path — `threatrag ingest --embedder gtr-base` — would have written
GTR-only points over all 40815 chunks and destroyed the MiniLM index that
M1–M4's numbers rest on. Confirmed against a live Qdrant on a throwaway
collection: upserting one named vector leaves the point holding only that one.

The fix is `attach_vectors`, which uses `update_vectors` to merge into the
stored point and leave the payload and other vectors alone, and a guard that
refuses the destructive ingest and names the safe command. After backfilling,
all 334 sampled points carry both vectors, the collection still holds 40815
points, and retrieval is unchanged. This was the most valuable thing M5 found
before it measured anything.

## Attack 1 — reconstruction (GTR): run live on a local GPU

`threatrag invert prepare` samples, attaches GTR vectors and writes the
bundles. One split is a property of the experiment, not tidiness:

| file | contents | leaves the machine |
|---|---|---|
| `vectors.npy`, `vector_ids.json` | 334 × 768 floats and opaque chunk ids | yes — this is all an index thief holds |
| `truth.jsonl` | source text, TLP, secret terms | **no** — the answer key |

Handing the reconstruction step the source text would make any rate it produced
circular, so scoring happens locally against a key the inversion never sees.

The reconstruction runs in an isolated venv (`.venv-inv`, `scripts/vec2text_invert.py`).
`vec2text` is pinned against a 2024 `transformers` and declares no upper bound,
so installing it beside the project would resolve cleanly and then break
`sentence-transformers` at runtime — taking M1–M4 with it — and its corrector
runs tens of forward passes per vector with beam search, which is minutes per
vector on CPU. So the working stack keeps its own environment and imports no
part of `vec2text`.

### The near-miss that would have invented a result

The first export was in **the wrong vector space**, and nothing would have said
so. The bundle was embedded with the `sentence-transformers` pipeline — mean
pool, then a 768→768 Dense projection, then L2 normalisation. But
`jxm/gtr__nq__32__correct` was trained against vec2text's own `gtr_base`
embedder, which is `AutoModel(...).encoder` → `last_hidden_state` → masked
mean, **and nothing else**.

Same model id, same 768 dimensions, no error anywhere. Measured on the live
model, the cosine between the two encodings of one sentence is **0.018** —
effectively orthogonal. The corrector would have received noise and returned
fluent nonsense, which is indistinguishable from an encoder that resists
inversion. M5 would have published a false negative and called it a finding.

`gtr-base` now resolves to a `MeanPooledEncoderEmbedder` that transcribes the
reference recipe; checked against vec2text's own code on the live model,
maximum absolute difference **0.0**. The retrieval encoder is untouched —
`all-minilm` is still `sentence-transformers`, so no M1–M4 number moves.

### Four silent transforms, all pointing the same way

That was not the only one. Getting a stolen vector from this index into a
public corrector meant crossing four convention boundaries, and **not one of
them raises an error when it is wrong**:

| # | mismatch | effect if missed | how it was caught |
|---|---|---|---|
| 1 | sentence-transformers pooling vs. raw mean pool | cosine **0.018** — orthogonal input | read vec2text's source instead of trusting the model id |
| 2 | 512-character chunks vs. a 32-token corrector | out-of-distribution input | read the checkpoint's `max_seq_length` |
| 3 | Qdrant normalises on write under cosine distance | magnitudes destroyed | norms came back exactly 1.000; verified with a norm-5 probe |
| 4 | `tokenize → decode` is lossy in sentencepiece | bundle encodes different text than the index | 26/334 cosines below 0.999 where all should have been 1.0 |

Every one of them degrades the reconstruction. Every one of them therefore
pushes the result toward "GTR resists inversion" — a *comfortable* conclusion,
arrived at by accident, that would have been indistinguishable from a real
finding.

The general lesson is worth more than any of the fixes: **when an attack and a
target are wired together only by a float array, mismatches fail silently and
they fail flattering.** A negative security result is only worth as much as the
plumbing checks behind it, so each link here was verified against ground truth
before any number was produced — max absolute difference 0.0 against vec2text's
own code, and cosine 1.000000 between the stored and unnormalised bundles once
the round trip was removed.

The fourth is the one that nearly slipped through, because it looked like
rounding. The first hypothesis — float noise from dynamic padding — was wrong,
and testing it said so (cosine 1.000000 with and without padding). Chunks whose
token vectors nearly cancel have a near-zero mean, so a small absolute error
there is a large relative one; the low-norm vectors were the visible casualties
of a fault affecting all of them.

### The storage layer is quietly destroying information too

A Qdrant collection using **cosine distance normalises vectors on write**.
Verified against a live instance: store `[3, 4, 0]`, read back `[0.6, 0.8, 0]`;
the same collection built with dot distance returns it intact.

So an attacker who steals *this* index gets directions, not magnitudes — and
the corrector was trained on unnormalised embeddings. That is a third silent
transform sitting between the attack and its target, after the wrong pooling
and the wrong sequence length.

It is a genuine property of the deployment, not a mistake to correct, so the
realistic bundle keeps it. But it should not be sold as a defence either: it is
an incidental side effect of a distance-metric choice, it costs the attacker
magnitude only, and any index built with dot distance gives it up for free.

### Three bundles, because a weak result has three possible causes

The corrector was trained on **32-token** passages and these chunks are 512
characters, so a weak reconstruction could mean *the attack does not work*,
*the text is longer than the attack was built for*, or *the magnitudes are
gone* — which imply very different things for a defender. Left confounded, the
result could be argued either way. Each bundle removes one variable:

| bundle | length | magnitudes | measures |
|---|---|---|---|
| `data/inversion/` | full chunk | normalised by Qdrant | the realistic attack on the index as it stands |
| `data/inversion/control/` | first 32 tokens | preserved | the corrector's own training conditions — the upper bound |
| `data/inversion/unnormalized/` | full chunk | preserved | isolates lost magnitude from chunk length |

Every manifest records its `variant`, because a reconstruction rate is
uninterpretable without knowing which bundle produced it.

The 32-token bundle's answer key holds the *decoded prefix*, not the full
chunk, and drops secret terms that fall past the budget: scoring a 32-token
vector against 512 characters would mark the reconstruction wrong for omitting
words its vector never carried.

**The gaps between the three are the finding.** If the control inverts well and
the stored bundle does not, then chunk length and cosine normalisation are
doing the defending — accidentally — and that is something M6 can turn into a
deliberate, measurable control.

### Result: 334 vectors per bundle, run on a local RTX 4060

1002 reconstructions, 20 correction steps with sequence beam width 4, ~124
minutes total. Per-chunk scores in `reports/data/m5_inversion_*.json`.

| bundle | exact | token F1 | BLEU-4 | round-trip cosine | note secrets |
|---|---|---|---|---|---|
| **stored** — full length, Qdrant-normalised | 0.024 | 0.254 | 0.0035 | 0.771 | 0.065 |
| **unnormalized** — full length, magnitudes | 0.117 | 0.328 | 0.0082 | 0.794 | 0.117 |
| **control** — 32 tokens, magnitudes | **0.222** | **0.740** | **0.4436** | **0.960** | **0.400** |

Both variables are real and they are separable:

- **Magnitude is worth about 5× exact match.** Restoring it (stored →
  unnormalized) moves exact match 0.024 → 0.117 and note secret recovery
  0.065 → 0.117, with length held constant. Qdrant's cosine normalisation is
  doing genuine defensive work — accidentally.
- **Length dominates text fidelity.** Shortening to the corrector's training
  window (unnormalized → control) moves token F1 0.328 → 0.740 and BLEU
  0.0082 → 0.4436, a 54× change, with magnitude held constant.

### What actually leaked

At the corrector's own training length, **16 of 34 confidential note chunks
gave up at least one declared secret, and 10 were fully reconstructed**. Two
TLP:RED passages came back **verbatim**:

> *"Entry Spearphishing attachment: a macro-enabled spreadsheet themed as a
> quarterly reconciliation, addressed to three named recipie…"*

> *"Impact and unreported detail The actor accessed but did not modify the
> payment approval matrix. Because the matrix documents which…"*

The second is the same secret M4's exfiltration attack went after. M4 needed a
poisoned document, a compliant generator and a rendered URL to get it out. Here
it falls out of a stolen 768-float array with no query, no model and no user.

Others degrade but still disclose: `43 seconds` (a beacon interval) survives
into a sentence whose other numbers are scrambled; an AMBER note returns *"used
a signed driver to obtain a handle to LSASS, and then wrote the dump to"* against
a truth of *"used a signed but outdated driver to obtain a handle to LSASS, then
wrote the dump to"*.

On the **stored** bundle — the realistic case — verbatim recovery collapses
(exact 0.000 for notes) but disclosure does not reach zero: 0.065 of declared
secrets and 2 fully leaked chunks. Reconstructions there read as topic-accurate
noise, recovering *LSASS*, *Windows*, *Temp* and a mangled `TREAS-W7734` while
losing the sentences around them.

### How to read this honestly

**This is not "GTR is invertible" and not "GTR is safe".** The measured claim
is narrower and it is the one the numbers support: *with the only publicly
available corrector, an attacker who steals this index recovers gist and
scattered identifiers, and would recover verbatim confidential text if the
chunks were short and the magnitudes intact.*

Three caveats bound it, and all three favour the defender:

1. The corrector is trained on 32-token Natural Questions passages — general
   web prose, not incident reports. An attacker who trained one on threat-intel
   text at 512 tokens would do better, and that is a few GPU-days, not a
   research programme.
2. The 0.024 exact match on the stored bundle is a floor produced partly by
   *this deployment's* accidental properties, not by GTR resisting inversion.
3. Nothing here was tuned. Steps and beam width are the published defaults, held
   identical across bundles so the comparison means something.

The defensive reading is the useful one: **long chunks and stored normalisation
each cost the attacker a large factor, and neither was chosen as a defence.**
M6 can make both deliberate and measure what they cost retrieval.

## Attack 2 — re-identification (MiniLM): the corrector-free attack, run live

No public corrector exists for `all-MiniLM-L6-v2`, the model the system
actually retrieves with. That is the honest negative result this milestone was
scoped around, and training one is the attacker's real cost — a corpus plus
multiple GPU-days.

Reporting only that would overstate it. An index that cannot be inverted is not
thereby safe. Most of this corpus is *published*: MITRE ships ATT&CK, NVD ships
CVEs, and the checkpoint is public too. So the attacker rebuilds the public
corpus himself — 6927 documents here — embeds it, and matches each stolen
vector to its nearest public document. Nothing is reconstructed. The vector is
recognised.

The reference is built from whole published **documents**, not this system's
chunks. The attacker has the publications, not the ingest configuration, and
matching stolen chunk vectors against the chunks they came from would measure
the identity function and report it as an attack.

### Result

| population | n | measure | rate |
|---|---|---|---|
| public chunks (ATT&CK, CVE) | 280 | matched to their own source document | **0.382** (107/280) |
| hidden chunks (notes, vendor) | 54 | subject disclosed by the neighbour's name | **0.407** (22/54) |
| internal notes only | 34 | subject disclosed | **0.353** (12/34) |
| hidden chunks | 54 | neighbour's identifier quoted outright | **0.000** (0/54) |

**38% of public chunks are re-identified exactly, with no corrector and no
reconstruction** — the attacker learns which ATT&CK technique or CVE a stolen
vector encodes by rebuilding a corpus he was always entitled to.

> **A lower bound (added 2026-09-15).** This reference embeds each document
> whole, and MiniLM reads only its first 256 tokens, so a chunk from deep inside
> a long ATT&CK page has nothing to match. Splitting the reference into passages
> of at most 600 characters, the same 280 chunks are recognised at **181/280 =
> 0.646**: ATT&CK 55 to 131, NVD 52 to 50. The 12/34 note disclosure count is
> unchanged, though five of the twelve notes differ. The 10 vendor chunks
> counted in the hidden row below were public PDFs missing from the reference;
> with them added, all 20 sampled vendor chunks are recognised outright. See the
> correction in `m7_benchmark.md`.

The confidential notes are the finding. Nothing public matches them, so they
cannot be recognised at all — but their nearest public neighbour names what
they are about:

| note | TLP | nearest public document | cos |
|---|---|---|---|
| INT-2026-001#1 *"obtain a handle to LSASS, then wrote the dump"* | AMBER | T1003.001 LSASS Memory | 0.503 |
| INT-2026-002#1 *"Spearphishing attachment: a macro-enabled spreadsheet"* | RED | T1566.001 Spearphishing Attachment | 0.645 |
| INT-2026-008#0 *(insider review, code repositories)* | RED | T1213.003 Code Repositories | 0.533 |
| INT-2026-007#0 *"Four domains registered... combine our brand name"* | GREEN | T1583.001 Domains | 0.600 |
| INT-2026-015#0 | AMBER | T1590.005 IP Addresses | 0.615 |

An attacker who cannot read a RED incident report still learns it is about
spearphishing attachments, or about code repositories. That is disclosure
without inversion, and the ACL enforced at retrieval does nothing about it —
the access-control model governs answers, and this attack never asks a question.

### The metric that reported zero

The first run of this attack reported **0/54** disclosures while working
perfectly. The metric asked whether a hidden chunk quotes its neighbour's
identifier verbatim, and analysts write prose: the note that says "obtain a
handle to LSASS" never types `T1003.001`, though that is exactly what it
matched. That strict question *still* answers 0/54 against the live index, so
it is kept as a column rather than deleted — a defender running the obvious
metric would conclude nothing leaked.

The replacement matches content words from the neighbour's name, with the
identifier and the parenthesised kind stripped (titles are stored as
`T1003.001 LSASS Memory (technique)`, and matching the whole string would score
the word "technique" as disclosure). Bare numbers are dropped, or a note dated
2026-02-11 would match every CVE published in 2026.

**0.353 is a floor, not an estimate.** A CVE title reduces to no content terms
once its id and severity are removed, and 15 of the 34 notes have a CVE as
their nearest neighbour, so the metric cannot fire on them at all. Among the 19
notes whose neighbour is a named ATT&CK object, the disclosure rate is
**12/19 = 0.632**. It also misses true hits lexically: INT-2026-006, on
ransomware pre-positioning, matched `S1058 Prestige` — a ransomware family, and
plainly the right subject — and scores as a miss because the words differ.

## What this changes for M6 and M7

- **A defence at the retrieval boundary cannot touch this.** Both attacks
  operate on stored vectors and never issue a query, so TLP filtering, trust
  tiering and answer-level checks are all downstream of the breach. The
  defences that apply are different in kind: authentication on Qdrant,
  encryption at rest, and — the one with a measurable utility cost worth
  plotting — not co-locating confidential notes with a public corpus in a
  single collection.
- **The MiniLM identifier weakness has a second face.** Dense MiniLM cannot
  retrieve by identifier (recorded four times now), yet it matches a document
  to its published source 38% of the time. Weak at the task, strong at the
  fingerprint.
- **Chunk size is now a security parameter, not only a retrieval one.** M1
  chose 512 characters on recall evidence alone. That choice turns out to cost
  the attacker a 54× factor in BLEU, and M6 can price the trade-off in both
  directions for the first time.
- **The cosine/dot distance choice is a defence nobody made.** Switching the
  collection to dot distance to preserve magnitudes would hand an index thief a
  5× improvement in exact reconstruction, for no retrieval benefit that M1
  measured. Worth stating as a deployment rule.
- **M4 and M5 reach the same secret by different routes.** The payment approval
  matrix required a poisoned document, a compliant generator and a rendered URL
  in M4; in M5 it falls out of a stolen float array. A defence set that stops
  one does nothing about the other, which is exactly the axis M7 should plot.
