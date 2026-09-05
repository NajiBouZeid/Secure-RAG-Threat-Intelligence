# Secure Threat Intelligence RAG

A retrieval-augmented threat intelligence assistant over MITRE ATT&CK, CVE and vendor
threat reports — which is then **red-teamed, defended, and quantitatively benchmarked**.

Most RAG projects stop at "it answers questions." The interesting part is that a RAG
system in a security context introduces an attack surface *at the retrieval layer* that
input-layer defences never see: the attacker's payload does not arrive through the user's
prompt, it arrives through a document the system chose to retrieve. This project builds a
useful system, breaks it four ways, defends it five ways, and measures the trade-off.

> **Status:** M1 in progress — ingestion, indexing, retrieval and retrieval evaluation.
> See [Roadmap](#roadmap).

---

## The three phases

| Phase | What it does |
|---|---|
| **1 — Build** | A working assistant over ATT&CK + NVD CVEs + vendor threat report PDFs, with citations, TLP classification and role-based access control. |
| **2 — Attack** | Indirect prompt injection via document poisoning; rendering-based data exfiltration; retrieval poisoning (PoisonedRAG-style); embedding inversion. |
| **3 — Measure** | A repeatable benchmark: attack success rate per attack class, per defence, per model — plotted against the retrieval-utility cost each defence imposes. |

The headline result is the **defence-effectiveness vs. utility-cost curve**. A defence that
stops every attack by making retrieval useless is not a defence, and attack success rate
reported without a utility baseline cannot tell you the difference.

---

## Architecture

```
          ingest                      query
   ┌──────────────────┐        ┌──────────────────┐
   │ ATT&CK · NVD ·   │        │  question +      │
   │ vendor PDFs      │        │  principal       │
   └────────┬─────────┘        └────────┬─────────┘
            │ on_ingest defences         │
            ▼                            ▼
      chunking (3 strategies)      embed query
            │                            │
            ▼                            ▼
      embed (MiniLM │ GTR)  ──► Qdrant ◄── filtered search
            │                            │  (TLP + source scope,
            ▼                            │   enforced server-side)
      named-vector index                 ▼
                                  on_retrieve defences
                                         │
                                         ▼
                                  generation (Ollama)
                                         │
                                         ▼
                                  on_answer defences
```

Three deliberate choices, each defensible rather than incidental:

**No LangChain in the core.** The retrieval layer *is* the object of study. A framework's
`RetrievalQA` hides exactly the seam where indirect injection, retrieval poisoning and
access-control failures live. The retriever here is a few hundred lines we own and can
instrument.

**Qdrant over Chroma.** Two load-bearing features: *named vectors*, so one collection holds
both the MiniLM and GTR embedding of the same chunk (the inversion attack needs GTR — the
only strong encoder with a public vec2text corrector — while the system runs on MiniLM);
and *server-side payload filtering*, so access control runs inside the query. Filtering
after retrieval is still wrong: a restricted chunk that displaces a permitted one changes
the top-k even after it is dropped.

**Ports and adapters.** `Embedder`, `VectorStore`, `Generator`, `Chunker` and `Defense` are
Protocols in `domain/ports.py`; concrete classes are chosen only in `factory.py`. That is
what makes the Phase 3 sweep (models × attacks × defence sets) a config matrix instead of
forked scripts.

**TLP and trust tiers exist from M1, not M6.** They are separate axes on purpose: TLP
governs *who may read* a document, trust tier governs *how much that document's content may
steer an answer*. Indirect prompt injection is fundamentally a trust-tier failure, and
cross-context exfiltration is meaningless to demonstrate against a system with no
classification model to violate.

---

## Quickstart

Requires Python 3.11+, Docker, and [Ollama](https://ollama.com) on the host.

```bash
make install                      # editable install with dev + pdf extras
make up                           # start Qdrant on :6333
ollama pull qwen2.5:7b-instruct   # generation model (M2 onward)

make fetch                        # ATT&CK (~40 MB), NVD CVEs, vendor PDFs (~73 MB)
make ingest                       # parse, chunk, embed, index
make query Q="What persistence techniques does APT29 use?"

python -m threatrag.cli build-goldset
make eval-retrieval               # Recall@k / MRR / nDCG
```

An NVD API key is optional. Without one the CVE fetch throttles to the public rate limit
and takes roughly forty minutes instead of five; with one, put it in `.env` as
`NVD_API_KEY=...` (gitignored, and never read from the committed config).

Ollama runs on the host rather than in Compose: GPU passthrough to a containerised Ollama
on Windows is unreliable and buys nothing. The API container reaches it through
`host.docker.internal`.

---

## Layout

```
configs/          base.yaml + experiment overlays (deep-merged)
src/threatrag/
  domain/         models and Protocol ports — depends on nothing
  ingest/         sources, chunking strategies, pipeline
  index/          embedders, Qdrant adapter
  rag/            retriever, prompting, generation
  security/       attacks/ and defenses/
  eval/           metrics, gold set, benchmark runner
  api/            FastAPI + demo UI
attacks/          versioned attack corpus (YAML)
tests/
```

## Roadmap

- [x] **M1** — ATT&CK ingest, chunking strategies, Qdrant index, retriever, Recall@k harness
- [x] **M2** — Generation with citations, FastAPI + demo UI, ACL wired end-to-end
- [x] **M3** — NVD CVE and vendor PDF sources, chunking-strategy comparison
- [x] **M4** — Indirect injection, rendering exfiltration, retrieval poisoning
- [ ] **M5** — Embedding inversion (GTR via vec2text; honest negative result for MiniLM)
- [ ] **M6** — Five defences, independently toggleable
- [ ] **M7** — Benchmark runner, two models, defence-vs-utility report

## Scope and ethics

`attacks/` contains working prompt-injection and poisoning payloads. They exist to
benchmark a **locally hosted, self-owned** system and are stored as inert data files that
nothing executes automatically. Nothing here targets a third-party service.

## Data sources and attribution

This product uses the NVD API but is not endorsed or certified by the NVD.

CVE records are reproduced with their descriptions verbatim; only enumerated lists are
bounded, and every truncation is marked in the document text. Each document links its
canonical NVD record.

Vendor threat reports remain the copyright of their publishers. `corpora/vendor_reports.yaml`
is a manifest of public URLs with hashes, not a redistribution: the PDFs are fetched at
build time and never committed.

MITRE ATT&CK(R) is a registered trademark of The MITRE Corporation.

## License

MIT — see [LICENSE](LICENSE).
