# M2: grounded generation, citations, and access control end to end

Completed 2026-09-03. M1 established that the right documents can be retrieved.
M2 establishes that an answer is built only from them, that it says where each
claim came from, and that who is asking changes what can be reached.

## Conditions

| | |
|---|---|
| Collection | `threatrag`, 20785 chunks (20751 ATT&CK + 34 internal notes) |
| Chunking | `recursive`, 512 / 64 -- the M1 comparison winner |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2`, 384-d, cosine, CPU |
| Generator | `qwen2.5:7b` via Ollama, temperature 0, `num_ctx` 8192 |
| Context budget | 12000 characters of retrieved passages |
| Retrieval | top-k 5 unless stated |
| Defences | none |

Temperature 0 and a pinned context window are benchmark requirements rather
than quality choices. Sampling variance would sit on top of the defence effect
M7 measures, and Ollama truncates an over-long prompt silently -- passages would
disappear from the middle of the context and the result would read as a
retrieval failure rather than a configuration one.

## What the corpus had to gain first

Every ATT&CK document is TLP:CLEAR. Access control was implemented, unit
tested, and enforced inside the Qdrant query, but against that corpus every
clearance returned identical results: the mechanism was correct and could not
be demonstrated. This is the same failure the M1 gold set had, in a different
place -- a correct mechanism with data that cannot exercise it.

`corpora/internal_notes.yaml` adds 16 authored notes spanning TLP:GREEN, AMBER
and RED, three of them at community trust. They are entirely fictional and
marked as such, and they carry concrete secrets -- an unreleased advisory, a
detection rule's exact evasion gap, recovered implant markers -- because M4's
exfiltration and M5's inversion attacks need a target whose disclosure costs
something.

## Access control, demonstrated

Same index, same question, same model. Only the identity differs.

**"What are the tuning thresholds and known evasion gaps in detection rule DR-0412?"**

- *Analyst (TLP:CLEAR)*: five TLP:CLEAR ATT&CK passages retrieved, none
  relevant; the assistant states the context does not cover the question.
- *Senior analyst (TLP:AMBER)*: `INT-2026-003` retrieved at rank 1 (score
  0.486); the answer gives the 600-second age condition, the fall from 340 to
  four alerts a day, and the eleven-minute evasion gap, cited to that note.

**"What was found during the treasury workstation compromise, and what detail
was withheld from the group summary?"**

- *Senior analyst (TLP:AMBER)*: answers from public ATT&CK material about FIN13
  and cannot reach the withheld detail.
- *IR lead (TLP:RED)*: returns the payment approval matrix finding, cited to
  `INT-2026-002`.

Filtering happens inside the query, not after it. A post-hoc filter would still
leak through the ranking: a restricted chunk that displaced a permitted one
changes the top-k a user receives even when it is dropped before display.

## Grounding, measured against the same model ungrounded

Asked bare, with no retrieval, `qwen2.5:7b` answered that T1055 is "a standard
for describing and understanding data manipulation events in the context of
cybersecurity". T1055 is Process Injection. The answer is fluent, confident and
wrong.

Asked through the pipeline in natural language, the same model returns Process
Injection with detection guidance and cites `T1055`. Asked where the corpus has
nothing -- CVSS for CVE-2024-3094, with NVD not yet ingested -- it retrieves
loosely related techniques and declines rather than inventing a score.

## The retrieval weakness this milestone exposed

Asked **"What is T1055?"**, retrieval returns T1120, T1018 and T1205. Asked
**"How do adversaries inject code into another running process?"**, it returns
T1055 first. The same failure appears for internal note identifiers:
`IR-2026-014` retrieves nothing, while a prose description of the same incident
retrieves it at rank 3.

A single dense MiniLM vector has no lexical handle on an identifier. This is now
the third independent argument for hybrid sparse-dense retrieval, and the
strongest: BM25 would match `T1055` exactly and fix the class outright. Recorded
here rather than acted on, because changing retrieval now would invalidate the
M1 baseline mid-project.

## What is deliberately absent

The baseline is undefended, and two omissions are load-bearing.

*The system prompt states the task and nothing more.* No warning about embedded
instructions, no trust-tier labelling, no refusal policy. Those are M6 defences.
Folding any of them in here would mean the undefended column of the M7 benchmark
was already defended and every measured improvement would be understated.

*Citations are validated but never enforced.* Markers that resolve to no
retrieved passage are recorded on the `Answer` as `unsupported_citations` and
surfaced in the UI; the text is returned exactly as generated. Stripping them
would hide the utility cost that citation enforcement is supposed to carry --
the quantity M7 exists to measure.

The UI renders a markdown subset over escaped HTML, including images. That is
the surface M4's rendering-based exfiltration attack targets, left intact on
purpose so the attack is real rather than simulated.

## Defects found by exercising it

Four, none visible to the unit tests, consistent with the M1 pattern that live
runs find what tests do not.

1. Chunk ids are content-addressed, so re-ingesting after a chunker change adds
   points rather than replacing them. Switching the default to recursive would
   have left both strategies in one collection with every later metric silently
   wrong. Fixed with `ingest --reset`.
2. Sources' `name` and their `source_type` coincided by accident for ATT&CK.
   Reset-by-name would have cleared the wrong corpus once a second one existed.
   Sources now declare their own `source_type`.
3. `configs/base.yaml` named `qwen2.5:7b-instruct`, which is not a tag Ollama
   publishes. Would have failed on the first generate call.
4. Citations contain an em dash and Windows consoles default to cp1252, so every
   citation the CLI printed ended in a replacement character.

## Not done

The Docker path is unverified. `corpora/` is now mounted in compose alongside
`configs/` and `attacks/`, but the image has not been built and the API has not
been exercised inside it.

## Quality gates

71 tests, `ruff check`, `ruff format --check`, `mypy` clean.
