# Secure Threat Intelligence RAG: Technical Report

Naji Bou Zeid
September 2026

Companion documents: the [Project Report](project_report.md) explains the concepts, the threat model and the results for a non-specialist reader, and the [README](../README.md) is the installation and reproduction guide.

---

## Abstract

This report documents the implementation of the Secure Threat Intelligence RAG system: the toolchain and why each tool was chosen, the repository layout, the software architecture, every subsystem from ingestion to the web interface, the test and quality infrastructure, and the engineering defects found during development together with their fixes. The system is a Python 3.11 package (`threatrag`, about 4,700 lines of source code and 415 tests) built on Qdrant, sentence-transformers and a locally served Ollama model. It is organised as ports and adapters, so that the benchmark varies embedders, stores, generators, chunkers and defences purely through configuration. More than forty defects are recorded. Most were invisible to unit tests and appeared only against real data, and a significant share affected the measuring apparatus rather than the system under test.

---

## Table of contents

1. [Scope](#1-scope)
2. [Toolchain](#2-toolchain)
3. [Repository structure](#3-repository-structure)
4. [Software architecture](#4-software-architecture)
5. [Configuration system](#5-configuration-system)
6. [Ingestion subsystem](#6-ingestion-subsystem)
7. [Index subsystem](#7-index-subsystem)
8. [Retrieval and generation](#8-retrieval-and-generation)
9. [Security subsystem](#9-security-subsystem)
10. [Evaluation subsystem](#10-evaluation-subsystem)
11. [API and web interface](#11-api-and-web-interface)
12. [Quality engineering](#12-quality-engineering)
13. [Defects found and fixed](#13-defects-found-and-fixed)
14. [Measured performance](#14-measured-performance)
15. [Reproducibility controls](#15-reproducibility-controls)
16. [Known limitations and open technical items](#16-known-limitations-and-open-technical-items)
17. [Appendix A: command-line reference](#appendix-a-command-line-reference)
18. [Appendix B: evidence files](#appendix-b-evidence-files)

---

## 1. Scope

The system ingests four corpora into a vector index, answers questions with cited retrieval-augmented generation under TLP-based access control, runs a corpus of document-borne attacks and two stolen-index attacks against itself, applies any subset of six defences, and benchmarks the result. This report covers how all of that is built. Results appear here only where they motivated an engineering decision; the full results are in the Project Report and in `reports/`.

The attack harness is described as software: its file format, lifecycle and scoring. The content and design of individual attack documents are not described. Those are in `attacks/` and `reports/m4_attacks.md`.

---

## 2. Toolchain

### 2.1 Runtime dependencies

| Tool | Version pin | Role | Why this choice |
|---|---|---|---|
| **Python** | ≥ 3.11 | Implementation language | `StrEnum`, modern typing, and the ML ecosystem |
| **pydantic** | ≥ 2.7 | Domain models, config schema, validation | Validated, typed models; configuration errors fail at load, not mid-run |
| **pydantic-settings** | ≥ 2.3 | Environment-variable overrides | Service URLs and the NVD key come from the environment, never from committed YAML |
| **PyYAML** | ≥ 6.0 | Config, attacks, manifests, notes | Human-editable, diffable data files |
| **Typer** | ≥ 0.12 | Command-line interface | Typed options with generated help |
| **Rich** | ≥ 13.7 | Console tables, status, colour | Readable benchmark and ingest output |
| **httpx** | ≥ 0.27 | HTTP client for NVD, PDFs, Ollama | Timeouts and connection reuse |
| **tenacity** | ≥ 8.3 | Bounded retries | Retries with exponential backoff that eventually give up |
| **NumPy** | ≥ 1.26 | Vector arithmetic, bootstrap | Standard array library |
| **sentence-transformers** | ≥ 3.0 | MiniLM embedding (and PyTorch transitively) | Reference implementation of the production encoder |
| **qdrant-client** | ≥ 1.9 (1.19 used) | Vector store client | Named vectors, payload filters, sparse vectors, server-side fusion |
| **FastAPI** | ≥ 0.111 | Web API | Typed request and response models, reusing the domain models |
| **uvicorn** | ≥ 0.30 | ASGI server | Serves the API and static pages |
| **pdfplumber / pypdf** | ≥ 0.11 / ≥ 4.2 | PDF text extraction (optional `pdf` extra) | pdfplumber gives the per-character layout used for artefact cleaning |

### 2.2 Services

| Service | Version | Role | Deployment |
|---|---|---|---|
| **Qdrant** | 1.19.0 (Docker image pinned) | Vector database | Docker Compose on ports 6333 (HTTP) and 6334 (gRPC), persistent volume |
| **Ollama** | 0.3x client | Local LLM server | Runs on the host, not in a container: GPU passthrough to a containerised Ollama on Windows is unreliable |
| **qwen2.5:7b** | Ollama tag | Primary generator | Large enough to follow instructions, so an attack outcome is not confounded with incoherence |
| **qwen2.5:1.5b** | Ollama tag | Second generator | Second data point for model dependence; fits beside 7b on 8 GB VRAM |

The Qdrant image is pinned to match the client, because the client refuses a major-version mismatch and warns beyond one minor version.

### 2.3 Models

| Model | Dimensions | Use |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 384, L2-normalised, cosine | Production retrieval embedder; 256-token input limit |
| `sentence-transformers/gtr-t5-base` | 768, raw masked mean pool, unnormalised | Inversion target only; the one strong encoder with a public vec2text corrector |
| `jxm/gtr__nq__32` and `jxm/gtr__nq__32__correct` | n/a | vec2text inversion and corrector models (about 2.5 GB), plus `t5-base` |

### 2.4 Isolated inversion environment

vec2text 0.0.13 (2024, unmaintained) cannot share an environment with the project. It is pinned against a 2024 `transformers` with no upper bound, so installing it beside the project resolves cleanly and then breaks sentence-transformers at runtime. It therefore runs in a separate virtual environment, `.venv-inv`, with CUDA PyTorch and a pinned 2024-era stack: transformers 4.44.2, sentence-transformers 3.0.1, tokenizers 0.19.x, datasets < 3, accelerate < 1.0 and huggingface_hub < 1.0. The project package never imports vec2text. `threatrag invert prepare` writes the input files, the standalone script `scripts/vec2text_invert.py` runs in `.venv-inv`, and `threatrag invert score` reads the output.

### 2.5 Development tooling

| Tool | Role |
|---|---|
| **pytest**, **pytest-cov** | 415 tests; coverage in CI |
| **ruff** | Linting (rule sets E, F, I, UP, B, SIM, RUF) and formatting, line length 100 |
| **mypy** | Strict type checking over `src` and `scripts` |
| **hatchling** | Build backend; installable package with a `threatrag` console script |
| **Make** | Task shortcuts (`install`, `up`, `fetch`, `ingest`, `eval-retrieval`, `serve`, `test`, `lint`) |
| **Docker / Docker Compose** | Qdrant, and an optional API container |
| **GitHub Actions** | CI: install with CPU-only PyTorch, lint, type-check, test |

### 2.6 Hardware used for the published results

An ASUS laptop with an NVIDIA RTX 4060 Laptop GPU (8 GB VRAM), running Windows 11. The project virtual environment uses CPU PyTorch, and MiniLM embedding runs on CPU at about 53 passages per second. Ollama uses the GPU. CUDA PyTorch exists only in `.venv-inv`, for vec2text.

---

## 3. Repository structure

```
.
├── attacks/                      Attack corpus (inert YAML data)
│   ├── *.yaml                    Seven original attacks (the "n/7" corpus)
│   └── evasion/                  Two evasion variants and their README
├── configs/
│   ├── base.yaml                 Single source of truth for a run
│   └── experiments/              Overlays deep-merged over base.yaml
│       ├── chunking_*.yaml       M1 chunking comparison collections
│       ├── defense_*.yaml        One defence alone, all five, D5 build, D6 at vendor tier
│       ├── retrieval_hybrid*.yaml  The two hybrid collections
│       └── attacker_public_view.yaml  What an index thief holds once D5 is deployed
├── corpora/
│   ├── internal_notes.yaml       16 fictional TLP-classified notes (committed)
│   └── vendor_reports.yaml       Manifest of 10 PDF URLs and SHA-256 hashes (PDFs not committed)
├── data/                         Gitignored: raw downloads, gold set, inversion bundles
├── docs/                         Project and technical reports
├── reports/
│   ├── m1_baseline.md … m7_benchmark.md, hybrid_retrieval.md   Milestone reports
│   └── data/                     Committed evidence (JSON, JSONL, per-run attack logs)
├── scripts/
│   ├── vec2text_invert.py        Inversion step, run in .venv-inv
│   ├── hybrid_eval.py            Dense vs hybrid retrieval evaluation
│   ├── utility_report.py         Paired utility comparison over a sweep file
│   └── answer_length.py          Verbosity hypothesis test
├── src/threatrag/
│   ├── domain/                   Models, ports, type aliases; no adapter imports
│   ├── ingest/                   Sources, chunking, ingest pipeline
│   ├── index/                    Embedders, Qdrant store, segregated store, BM25
│   ├── rag/                      Retriever, answer pipeline, Ollama generator
│   ├── security/
│   │   ├── attacks/              Schema, loader, render, runner, evaluate, sink
│   │   ├── defenses/             Registry and six defences
│   │   └── inversion/            Sample, backfill, export, score, reidentify
│   ├── eval/                     Gold set, metrics, answer utility, benchmark,
│   │                             paired bootstrap, probes, findings reader
│   ├── api/                      FastAPI app and two static pages
│   ├── config.py                 Typed config schema and loader
│   ├── factory.py                The only place concrete classes are chosen
│   └── cli.py                    Typer command-line entry point
├── tests/                        415 tests, one module per component
├── Dockerfile, docker-compose.yml, Makefile, pyproject.toml
├── .env.example                  Template for local environment variables
└── .github/workflows/ci.yml
```

### 3.1 Package modules

| Module | Responsibility |
|---|---|
| `domain/models.py` | `TLP`, `TrustTier`, `SourceType`, `Document`, `Chunk`, `Principal`, `RetrievedChunk`, `Answer` |
| `domain/ports.py` | Protocols: `Embedder`, `Chunker`, `VectorStore`, `HybridSearch`, `Generator`, `DocumentSource`, `Defense` |
| `domain/types.py` | `Vector` and `Matrix` NumPy type aliases for strict typing |
| `config.py` | Pydantic schema for every config section; YAML loading, overlay merging, environment overrides |
| `factory.py` | Builds sources, chunker, embedders, stores, generator, defences and pipelines from a `Config` |
| `cli.py` | Commands: `fetch`, `ingest`, `build-hybrid`, `build-goldset`, `query`, `ask`, `eval-retrieval`, `status`, `attack`, `invert`, `benchmark` |
| `ingest/sources/base.py` | Shared HTTP client settings, project User-Agent |
| `ingest/sources/attack_cti.py` | STIX bundle parsing, detection join, procedure-example folding, CVE mention extraction, group-technique pairs |
| `ingest/sources/nvd_api.py` | NVD API 2.0 client: windows, paging, pacing, retries, cache |
| `ingest/sources/nvd_cve.py` | CVE selection and document shaping |
| `ingest/sources/vendor_report.py` | PDF download, hash verification, text extraction and cleaning |
| `ingest/sources/internal_notes.py` | Loads the authored notes with their TLP, tier and secret terms |
| `ingest/chunking.py` | Structural, recursive and fixed chunkers; content-addressed chunk ids |
| `ingest/pipeline.py` | Ingest-time defences, chunk, embed, upsert; statistics |
| `index/embedders/sentence_transformer.py` | MiniLM adapter |
| `index/embedders/mean_pooled.py` | GTR adapter matching vec2text's reference embedder exactly |
| `index/qdrant_store.py` | Qdrant adapter: collections, filters, upsert, attach, scroll, hybrid search, copy |
| `index/segregated_store.py` | Two-collection store for D5, merged by score |
| `index/sparse/bm25.py` | Tokeniser, stable term hashing, BM25 term-frequency weights |
| `rag/retriever.py` | Embed, search with principal, over-fetch, retrieval-time defences, truncate |
| `rag/pipeline.py` | System prompt, context formatting, generation, citation resolution, answer-time defences |
| `rag/generators/ollama.py` | Ollama HTTP generator with pinned options, retries and a load-state probe |
| `security/attacks/*` | Attack schema, loader, placeholder rendering, runner, success evaluation, exfiltration sink |
| `security/defenses/*` | Registry and the six defences |
| `security/inversion/*` | Sampling, vector backfill, bundle export, reconstruction scoring, re-identification |
| `eval/goldset.py` | Gold set built from ATT&CK relationships |
| `eval/metrics.py` | Recall, precision, MRR, nDCG, hit rate at document level |
| `eval/answers.py` | Judge-free answer utility, answer recall, ungrounded identifier rate |
| `eval/benchmark.py` | Cell construction, resumable JSONL writer, attack and utility routes |
| `eval/paired.py` | Paired percentile bootstrap over questions |
| `eval/probes.py` | Identifier probe sets and their name-phrased twins |
| `eval/findings.py` | Reads committed evidence into ranges for the findings page |
| `api/main.py` | FastAPI routes, service container, attack runner lock |
| `api/static/index.html` | Console page (question, principal, retriever, defence toggles, audit panels) |
| `api/static/findings.html` | Findings page rendering committed evidence |

---

## 4. Software architecture

### 4.1 Ports and adapters

Every component the benchmark varies is a `typing.Protocol` in `domain/ports.py`, and nothing outside `factory.py` names a concrete class. The domain package imports no adapter. This buys three things:

- a benchmark cell is a configuration, not forked code;
- tests substitute in-memory fakes for Qdrant and Ollama, so the suite needs no running services;
- the CLI, the API and the benchmark build their objects through the same factory, so they cannot disagree about what a configuration means.

Two ports carry non-obvious contracts.

- **`VectorStore.upsert` replaces points.** In Qdrant, upserting a point that carries one named vector drops every other named vector on that point. A second encoder's vectors must therefore be added with `attach_vectors`, which uses Qdrant's `update_vectors`. This contract was discovered the hard way (Section 13.5).
- **`HybridSearch` is a separate protocol.** A lexical ranking needs the question text, which no other store requires, so widening `search` would impose a parameter on every adapter. The retriever asks the store whether it is hybrid rather than inspecting its type.

### 4.2 Domain model

`Document` and `Chunk` carry `tlp`, `trust_tier` and `source_type` from ingestion onward, and `Chunk` denormalises them from its parent, so Qdrant can filter on them server-side. `Principal.may_read(chunk)` compares TLP rank with clearance and checks an optional allowed-source-type set.

`Answer` is also an audit record. Alongside the text it holds:

- the retrieved passages;
- resolved citations and `unsupported_citations`;
- `stripped_urls` (from D3);
- `defenses_applied` and `defenses_abstained` (from D6);
- `blocked` and `block_reason`.

These fields exist so the benchmark can count outcomes a defence would otherwise hide. An empty sink log, for example, cannot distinguish a blocked exfiltration from one never attempted, but `stripped_urls` can.

### 4.3 Data flow

**Ingestion.** `DocumentSource.load()` yields `Document`s, and each ingest-time defence may reject a document (`on_ingest` returns `None`). The `Chunker` produces `Chunk`s, and the `Embedder` produces a `Matrix`. `VectorStore.upsert` then writes the points.

**Query.** `Retriever.retrieve` embeds the question and searches with the principal's filter at depth `top_k × overfetch`, optionally applying a score threshold. It re-checks `may_read` as a safeguard and applies each `on_retrieve` defence in registry order, then truncates to k. `AnswerPipeline.answer` formats the context, generates, resolves citations and applies each `on_answer` defence.

### 4.4 Defence registry

`security/defenses/__init__.py` registers the six defences in a fixed canonical order: `injection_screen`, `source_cap`, `provenance_fence`, `egress_filter`, `corroboration`, `corpus_segregation`. `build_defenses(config)` instantiates the enabled ones in that order, whatever the order in the config, so `{a, b}` and `{b, a}` are the same cell. An unknown name raises `ValueError` listing the available names.

---

## 5. Configuration system

### 5.1 Base configuration and overlays

`configs/base.yaml` is the single source of truth for a run. Its sections are:

- `paths`;
- `embedding` (model registry and primary);
- `chunking` (strategy, size 512, overlap 64);
- `vector_store` (URL, collection, and for D5 a restricted collection and threshold);
- `retrieval` (top-k 5, score threshold, over-fetch, mode, BM25 parameters, hybrid prefetch);
- `generation` (Ollama URL, model, temperature 0, `num_ctx` 8192, `num_predict` 1024, `keep_alive` 60m, context budget 12,000 characters);
- `principals` (three demo identities);
- `sources` (per-corpus settings);
- `defenses` (membership list);
- `defense_settings` (per-defence parameters).

Experiment overlays in `configs/experiments/` are deep-merged over the base, so an overlay states only what it changes. The `--overlay/-o` option is repeatable, which lets a hybrid retriever and a defence set compose in one run. Every benchmark cell is identified by its overlay file.

### 5.2 Validation

The whole config is a pydantic model, so a typo or a wrong type fails at load. Cross-field validators refuse combinations whose results would be meaningless:

- hybrid retrieval together with `corpus_segregation`, because the segregated store merges by score, which is exact for cosine similarity and meaningless for RRF scores;
- hybrid retrieval together with a score threshold, for the same reason.

The API rebuilds a per-request config with `Config.model_validate` rather than `model_copy`, so these validators also run for requests, and a forbidden combination returns HTTP 400 with the config layer's own message.

### 5.3 Environment

`pydantic-settings` reads `THREATRAG_QDRANT_URL`, `THREATRAG_OLLAMA_URL`, `THREATRAG_CONFIG`, `THREATRAG_DATA_DIR` and `NVD_API_KEY` from the environment or from a gitignored `.env`. The NVD key is never read from committed configuration, and it is sent only in a request header, never in a URL that could be logged.

---

## 6. Ingestion subsystem

### 6.1 MITRE ATT&CK (`attack_cti.py`)

- **Source:** `mitre-attack/attack-stix-data`, the Enterprise bundle (release 19.2, 26,086 objects, 53.8 MB). The older `mitre/cti` mirror ships a smaller bundle; the corpus was fixed before any baseline was recorded.
- **Object kinds indexed:** techniques and sub-techniques (697), software (825), groups (176), mitigations (44) and tactics (15), giving 1,757 documents. Revoked and deprecated objects are excluded. Campaign objects are not indexed as documents (Section 16).
- **Detection join.** From ATT&CK v18 the `x_mitre_detection` field is empty on every technique. Detection guidance lives in `x-mitre-detection-strategy` objects linked by `detects` relationships (one per technique), which reference `x-mitre-analytic` objects carrying the description and log-source references. These are joined back into each technique document as a detection section.
- **Procedure examples.** All 17,136 `uses` relationships targeting attack patterns carry a description that names the actor. They are folded into the target technique's document, attributed to the source group or software. Without this, gold-set evidence coverage is 5/941; with it, 941/941.
- **CVE mentions.** ATT&CK has no structured CVE field. `cve_mentions()` applies a CVE regular expression over `description` fields only, the fields actually indexed, and attributes a relationship's mention to both endpoints. This yields 171 CVEs and about 479 links. It feeds the NVD selection and is never used as gold labels.
- **Gold-set feed.** `group_technique_pairs()` exposes the curated group-uses-technique edges with tactic information.

### 6.2 NVD (`nvd_api.py`, `nvd_cve.py`)

- **API client.** NVD CVE API 2.0. Publication-date windows are capped at 120 days and pages at 2,000 results. Requests are paced at the published interval (6 seconds without a key, 0.6 seconds with one) with no concurrency. Retries are bounded, honour `Retry-After`, and then give up. The API key travels in the `apiKey` header. Responses are cached per request under `data/raw/nvd/`, so raising a limit refetches only the difference.
- **Selection.** First, every ATT&CK-linked CVE (170 with NVD records; one cited CVE has none, which is treated as data, not failure). Second, CRITICAL and HIGH CVEs walked backwards over 18 months in 120-day windows from the fixed `window_end: 2026-09-01`, capped at 5,000. That gives 5,170 documents in total.
- **Tombstones.** Records whose `vulnStatus` is Rejected, or whose description begins with `** REJECT`, are skipped.
- **Document shape.** The description is kept verbatim. The primary CVSS metric is taken from the newest version available (v4.0, then v3.1, v3.0, v2.0), preferring NVD's own Primary analysis over a vendor Secondary one within a version. The document adds CWE identifiers, affected products (bounded lists, with truncation marked in the text) and a reference summary by tag with a count, instead of the URL list. The canonical NVD URL is kept on the document.
- **Terms of use in code.** The pacing, retry, header and attribution obligations are asserted by tests in `tests/test_nvd_api.py`, because the most likely future edit to that module is an attempt to make it faster.

### 6.3 Vendor PDFs (`vendor_report.py`)

- **Manifest.** `corpora/vendor_reports.yaml` lists ten reports with id, publisher, date, URL, provenance (`publisher` or `mirror`), filename and SHA-256. The hash is recorded on first fetch and verified on every later fetch, so a rotted URL that now serves a login page named `.pdf` fails loudly.
- **Download.** A descriptive project User-Agent is sent. One CDN returns 403 to httpx's default User-Agent; the project identifies itself rather than impersonating a browser.
- **Extraction floor.** A document with fewer than 200 extractable characters per page is rejected. This caught a fully rasterised report.
- **Cleaning.** Three artefacts are removed:
  - faux-bold doubling (`TTHHRREEAATT`), collapsed per line only when every alphanumeric character is paired, so words like "committee" survive;
  - table-of-contents dot leaders;
  - running headers and footers, detected by frequency after normalising digits, because the page number changes on every page.
- **Not fixed:** multi-column interleaving. pdfplumber reads across columns in one report.

### 6.4 Internal notes (`internal_notes.py`)

The notes are loaded from `corpora/internal_notes.yaml`: 16 fictional notes with TLP, trust tier, ATT&CK technique tags and declared `secret_terms`. They are the only corpus stored in the repository, and they are enabled by default, because without them every document is TLP:CLEAR and the access-control path cannot be exercised.

### 6.5 Chunking (`chunking.py`)

All three strategies implement the `Chunker` protocol, with a chunk size of 512 characters and an overlap of 64.

| Strategy | Behaviour |
|---|---|
| `structural` | Splits on the `## ` section headings the sources emit, then bounds size |
| `recursive` (default) | Descends the separator hierarchy `"\n## "`, `"\n\n"`, `"\n"`, `". "`, `" "`; a piece that is still too large is re-split with the *next finer* separator |
| `fixed` | Fixed-width windows with overlap, ignoring structure |

**Chunk ids are content-addressed:** `"{doc_id}#{ordinal}:{sha1(text)[:12]}"`. Re-ingesting unchanged text reuses the same point, and edited text cannot leave stale chunks at the same id. Two consequences follow. Changing the chunker adds points rather than replacing them, hence `ingest --reset` (Section 13.2). And any text that varies from run to run inside an indexed document changes its id and its embedding, which is what made an ephemeral port number a source of variance (Section 13.7).

### 6.6 Ingest pipeline (`pipeline.py`)

For each document: apply `on_ingest` defences, chunk, embed in batches, upsert, and count documents seen, rejected and chunks indexed. Under the D5 overlay, the store is a `SegregatedStore` that routes each chunk by TLP.

The CLI adds two guards around the pipeline:

- **Refuse.** It will not ingest a populated collection under a non-primary embedder, since the upsert would drop the primary vectors.
- **Warn.** It warns, before and after, when re-ingesting under the primary embedder will drop secondary vectors previously attached by `invert prepare`, and names the command that repairs this.

---

## 7. Index subsystem

### 7.1 Collection design (`qdrant_store.py`)

- **Named vectors.** Every collection is created with both `all-minilm` (384, cosine) and `gtr-base` (768, cosine), because named vectors cannot be added to an existing collection. Points may carry a subset. In `threatrag`, all 40,815 points carry `all-minilm`, and the 334 inversion-sample points also carry `gtr-base`.
- **Payload.** Chunk text, title, `doc_id`, ordinal, `source_type`, `source_ref`, URL, `tlp`, `trust_tier` and metadata.
- **Point ids.** Qdrant requires UUIDs or integers, so chunk ids are mapped deterministically to UUIDs (UUIDv5), and the chunk id is also kept in the payload.
- **Access filter.** `search` builds a Qdrant filter matching any TLP value the principal's clearance can read, plus an optional source-type match, and passes it inside the query. The retriever re-checks `may_read` afterwards, so an adapter that ignored the principal could not become a bypass.
- **Other operations:**
  - `attach_vectors` adds one named vector to existing points (`update_vectors`), skipping unknown ids;
  - `get_vectors` reads vectors back, which is the attacker's view;
  - `scroll_chunks` enumerates passages, optionally for one source type;
  - `count_with_vector` counts points that carry a given named vector, the only evidence a backfill is intact;
  - `delete_by_source_type` supports `--reset` and attack cleanup;
  - `copy_from` supports the hybrid build.

### 7.2 Collections in use

| Collection | Points | Purpose |
|---|---|---|
| `threatrag` | 40,815 | Production and dense control |
| `threatrag_chunk_structural` / `_fixed` / `_recursive` | 19,833 / 14,836 / 20,751 | M1 chunking comparison (ATT&CK only) |
| `threatrag_public` / `threatrag_restricted` | 40,787 / 28 | D5 segregated layout (sums to 40,815) |
| `threatrag_hybrid` | 40,815 | Dense vectors plus BM25 over title and text |
| `threatrag_hybrid_text` | 40,815 | Dense vectors plus BM25 over text only |

### 7.3 Segregated store (`segregated_store.py`)

On write, a chunk with TLP above `restrict_above` (GREEN by default) goes to the restricted collection, and everything else to the public one. On search, both are queried with the same principal filter for k results each, and the two lists are merged by score. With one embedder and one metric the scores are globally comparable, so any chunk in the global top k is in its own collection's top k, and the merge reproduces the single-collection result set. Tie ordering can differ, and this was measured as a fourth-decimal change in MRR and nDCG.

### 7.4 BM25 and hybrid search (`index/sparse/bm25.py`)

- **Tokeniser.** A single regular expression that claims identifiers first, so `CVE-2024-3094`, `CWE-426`, `T1055`, `T1055.001` and `TA0001`/`G0007`/`S0356`/`M1038`/`C0024` are each one token. Identifiers must stand alone, so `xt1055` and `T10555` remain ordinary words. Everything else is split on punctuation and lowercased. A small stopword list removes question phrasing ("which", "does", "use").
- **Term ids.** A stable hash (a 4-byte BLAKE2b digest), not Python's `hash()`, which is salted per process and would make a collection built in one run unqueryable in the next.
- **Weights.** The sparse vector carries only the term-frequency part, `tf·(k1+1) / (tf + k1·(1 − b + b·len/avg_len))`, with k1 = 1.2 and b = 0.75. The collection is created with `Modifier.IDF`, so Qdrant computes IDF from the corpus it actually holds. That includes an attack document added later, which a build-time IDF would have excluded.
- **`avg_len`** is the measured mean BM25 token count per passage: 42.5 with titles, 38.9 without. It is stated in each overlay, and changing it requires a rebuild.
- **Hybrid query.** One Qdrant query with two prefetches, dense and sparse, each at depth 50, fused by RRF on the server. The access filter is applied to *both* prefetches; on the dense one alone, the lexical ranking would return passages the caller is not cleared for.
- **Deterministic ties.** RRF scores tie by construction. The adapter fetches all fused candidates, sorts by score and then chunk id, and cuts to k itself (Section 13.9).
- **Build.** `threatrag build-hybrid` copies points and dense vectors from `threatrag` (unchanged, bit-identical on a 1,000-point check) and adds sparse vectors. It refuses to build in place over the source and verifies the copied count. Each build takes about 40 seconds.

## 8. Retrieval and generation

### 8.1 Retriever (`rag/retriever.py`)

`Retriever` holds an embedder, a store, `top_k`, an optional `score_threshold`, `overfetch` (default 1) and the enabled defences. `retrieve` searches at depth `k × overfetch`, filters by threshold and `may_read`, applies `on_retrieve` defences, and only then truncates to k. The truncation comes after the defences deliberately. The over-fetched tail is what a diversity rule such as D2 promotes from, and the caller still receives k passages. With the default over-fetch of 1, every pre-M6 number is reproduced exactly.

### 8.2 Answer pipeline (`rag/pipeline.py`)

- **System prompt.** The prompt states the task and nothing more: answer only from the numbered passages, cite `[n]` after each claim, say plainly when the context does not answer, and prefer identifiers to general description. It carries no warning about embedded instructions, no trust labels and no refusal policy, because each of those is a defence, and folding one into the baseline would understate every measured improvement.
- **Context formatting.** Passages are numbered from 1 in rank order as `[n] {source_ref} — {title}` followed by the passage text, within a 12,000-character budget. A passage that would exceed the budget is dropped whole, never truncated, because half a passage consumes context and invites citation of evidence the reader cannot see.
- **Empty retrieval.** If nothing is retrieved, the pipeline returns a fixed "no evidence" answer without calling the model. An empty context is exactly the condition under which a model answers from memory.
- **Citation resolution.** `[n]` markers are extracted with a regular expression. Markers that map to a retrieved passage become `citations`; the rest become `unsupported_citations`. The text is returned unmodified. Citation enforcement is a defence whose cost must remain measurable.

### 8.3 Ollama generator (`rag/generators/ollama.py`)

The generator calls `/api/chat` with pinned options: temperature 0, `num_ctx` 8192, `num_predict` 1024, and `keep_alive` 60m. The context window is pinned because Ollama otherwise truncates an over-long prompt silently, dropping passages from the middle of the context; the resulting failure would read as a retrieval problem rather than a configuration one. The output bound exists because of the liveness defect in Section 13.8.

Other details:

- The HTTP timeout is 180 seconds.
- tenacity retries transport errors and timeouts up to three times with exponential backoff, but never HTTP status errors, which are deterministic.
- `load_state()` queries `/api/ps` for the loaded model's `size` and `size_vram` and reports whether it is fully on the GPU. The benchmark uses this to warn when layers are on the CPU (which changes the arithmetic) or when the model was reloaded mid-sweep.

---

## 9. Security subsystem

### 9.1 Attack harness (`security/attacks/`)

| Module | Role |
|---|---|
| `schema.py` | Pydantic model of an attack file: `id`, `family` (injection, exfiltration, poisoning), `description`, `target_query`, `clearance`, `doc` (title, `source_ref`, text, TLP, trust tier) and a list of success criteria. There is **no `source_type` field**: every attack document is `SYNTHETIC_ADVERSARIAL` by construction and cannot be ingested under another corpus's label. |
| `loader.py` | Loads `*.yaml` from a directory, non-recursively, so `attacks/` stays exactly seven attacks and the evasion variants in `attacks/evasion/` run only when asked for. |
| `render.py` | Substitutes the runtime address of the local sink for a placeholder in the document text. |
| `runner.py` | For each attack: render, ingest through the ordinary pipeline (including any ingest-time defences), ask the target query under a principal with the stated clearance, evaluate, and **delete the attack document in a `finally` block**. It records the retrieved passages by `doc_id` and `source_type`, the answer, the defences applied and abstained, stripped URLs and the sink log. |
| `evaluate.py` | Criteria types: `output_contains`, `output_matches` (regular expression), `retrieved_topk` (the attack's own document was retrieved) and `sink_received` (a logged request contained a value). An attack lands only if every criterion holds. For exfiltration, it requests the Markdown image and link URLs in the answer that target the local sink, and only those. This reproduces the rendering behaviour of the demo interface without contacting any other host. |
| `sink.py` | `ExfiltrationSink`: a loopback HTTP server that records every request's path and query string and returns a 1×1 GIF. It binds the fixed port 24601 (`DEFAULT_SINK_PORT`) with `SO_REUSEADDR`, so a sweep can reopen it once per cell, and it fails fatally if the port cannot be bound (Section 13.7). |

`threatrag attack clean` removes any residual `synthetic_adversarial` points, for example after an interrupted run. The index is verified to return to 40,815 points.

### 9.2 Defences (`security/defenses/`)

Each defence subclasses `BaseDefense`, whose hooks default to identity, and exposes a `build(config)` function.

| Defence | Hook | Implementation notes |
|---|---|---|
| `injection_screen` | `on_ingest` | 13 built-in case-insensitive patterns, plus configurable extras, over `title + text`. The title is included because it is denormalised onto every chunk. It never reads trust tier or source type. `matches()` is public so that the false-positive census can be run over the corpus. |
| `source_cap` | `on_retrieve` | Keeps at most `max_per_document` (2) passages per `doc_id`, preserving rank order. Its overlay also sets `retrieval.overfetch: 3`. |
| `provenance_fence` | `on_retrieve` | Wraps each passage at or above tier `min_tier` (VENDOR by default) in a quoted-material header and footer naming the tier. The rewritten text reaches both the prompt and the audit record, so the transcript shows what the model actually saw. |
| `egress_filter` | `on_answer` | Two regular expressions: Markdown links and images, and bare `http(s)` URLs not already inside Markdown. Non-allowlisted URLs are replaced by the link text, by `[image: alt]` for images, or by `[link removed]` for bare URLs, and each is recorded in `stripped_urls`. Allowlisting matches the exact host or a true subdomain, not a suffix, so that `evil-attack.mitre.org.example.com` is not accepted for `attack.mitre.org`. The default allowlist is empty. |
| `corroboration` | `on_answer` | If every cited passage is at or below the threshold tier (COMMUNITY by default), and an uncited AUTHORITATIVE passage has a `source_ref` equal to a cited passage's `source_ref` or contained in its text, the answer is replaced by a refusal with `blocked=True`. An answer with no citations, where low-trust material was retrieved, records an abstention in `defenses_abstained` rather than passing silently or refusing. |
| `corpus_segregation` | none (marker) | Its effect lies in the store the factory builds (`SegregatedStore`). The name exists so that it appears in `defenses_applied` and in cell names. |

### 9.3 Inversion and re-identification (`security/inversion/`)

| Module | Role |
|---|---|
| `sample.py` | Deterministic sample (seed 20260906): all internal-note chunks as a census, plus 300 others allocated across sources by largest remainder in proportion to index share. Attack documents are excluded. |
| `backfill.py` | Embeds the sampled chunks with GTR and attaches the `gtr-base` vector via `attach_vectors`, leaving `all-minilm` and the payload untouched. |
| `export.py` | Writes a bundle: `vectors.npy` and `vector_ids.json` (what a thief holds), `truth.jsonl` (the answer key: text, TLP and secret terms, which never goes to the reconstruction step) and `manifest.json` (variant, seed, model, counts). Three variants: `stored` (vectors read back from Qdrant, so normalised), `unnormalized` (full length, raw magnitudes) and `control` (first 32 tokens, raw magnitudes). The control answer key keeps the decoded 32-token prefix and drops secret terms beyond it. |
| `mean_pooled.py` (index) | `MeanPooledEncoderEmbedder`: the T5 encoder's `last_hidden_state`, masked mean pooling, no projection and no normalisation. It transcribes vec2text's reference `gtr_base` embedder; the maximum absolute difference against vec2text's own code was measured as 0.0. |
| `scripts/vec2text_invert.py` | Runs in `.venv-inv`. Loads the pretrained corrector and inverts each bundle with 20 correction steps and sequence beam width 4, the published defaults, held identical across bundles. It writes `reconstructions.jsonl`, and a `--limit` option supports smoke tests. It installs a stub for the Unix-only `resource` module on Windows. |
| `score.py` | Scores reconstructions against the answer key: exact match, token F1, corpus BLEU-4, round-trip cosine (re-embed the reconstruction and compare it with the stolen vector) and per-note secret-term recovery, broken down by source type and TLP. |
| `reidentify.py` | Builds the attacker's reference from public documents (ATT&CK and NVD by default; `--sources` adds vendor reports), optionally split into whitespace-bounded passages of at most `--passage-chars` characters, each mapped back to its document. Each stolen MiniLM vector is matched to its nearest reference vector. Public chunks are scored on recognition of their own document. Hidden chunks (notes, and vendor reports when absent from the reference) are scored on *subject disclosure*: whether content words of the neighbour's title, with the identifier, the parenthesised kind and bare numbers removed, appear in the hidden chunk. A strict *identifier quoted* column is kept alongside. `--dump` writes per-chunk rows as evidence. |

The reference must be built from whole published documents, not from this system's own chunks. Matching stolen chunk vectors against the chunks they came from would measure the identity function.

---

## 10. Evaluation subsystem

### 10.1 Gold set (`eval/goldset.py`)

The gold set is built from ATT&CK's curated `intrusion-set --uses--> attack-pattern` edges, grouped by (group, tactic). Groups with 3 to 8 techniques are kept, 200 are sampled with seed 1337, and the questions are phrased "Which *tactic* techniques does *group* use?". The set is stored at `data/eval/attack_goldset.json`. An LLM-generated question set was rejected, because questions written from the indexed text reward the retriever for surfacing the text they were copied from.

### 10.2 Retrieval metrics (`eval/metrics.py`)

Recall@k, precision@k, MRR, nDCG@k and hit rate. Retrieved chunks are collapsed to their document's `source_ref` before scoring, in first-occurrence order, so several chunks of one technique count once.

### 10.3 Answer utility (`eval/answers.py`)

- `answer_utility`: 1 if the answer names, in full, at least one technique id the gold set marks relevant. A bare parent id does not satisfy a sub-technique, and a refusal scores 0.
- `answer_recall`: the fraction of relevant ids named.
- `ungrounded_id_rate`: ids named that appear in no retrieved passage. This is scored against the evidence rather than the gold set, because MITRE's edges are incomplete.
- `refusal_rate` and `unsupported_citation_rate` are reported alongside.

Every question's record, including the full answer text, is persisted, so a later metric can be computed from the same answers.

### 10.4 Benchmark (`eval/benchmark.py`, `cli.py benchmark`)

- **Cells.** A cell is (defence-set overlay, model, repeat). Seven sets are defined: `none`, one per request-time defence, and `all`. `--sets` selects a subset, and `--models` takes a comma-separated list.
- **Routes.** The attack route runs the original corpus and the evasion corpus and records per-attack outcomes. The utility route answers `--questions` gold questions. Retrieval is scored once per defence set, since it does not involve the model. `--skip-attacks` runs utility only.
- **Resumable output.** Each finished cell is appended as one JSONL row to `--out` (default `reports/data/m7_sweep.jsonl`), and a re-run skips cells already present. `--fresh` discards the file. This exists because Ollama returned timeouts after about 75 minutes of continuous load.
- **`--repeats N`** runs each cell N times. It exposed the unstable attacks.
- **`--warm`** generates each question twice and keeps the second answer, recorded per row (Section 13.10). It costs about 75% more time, not double, because retrieval scoring is not repeated.
- **Load guard.** Before each model's cells, `check_model_load` issues a probe generation and inspects `/api/ps`. It warns on CPU layers or on a mid-sweep reload and records the state in the row.
- **Sink.** A single sink on port 24601 is reopened per cell.

### 10.5 Paired statistics (`eval/paired.py`, `scripts/utility_report.py`)

`paired_delta` computes a percentile bootstrap of the mean per-question difference between two cells, resampling questions with a fixed seed. `utility_report.py` pairs each cell with **every** undefended repeat of its model and prints the range of deltas and intervals. A change is reported as a finding only if every interval excludes zero on the same side. The undefended repeats are paired with each other in the same way, to show the noise floor.

### 10.6 Identifier probes (`eval/probes.py`, `scripts/hybrid_eval.py`)

Seeded sets of 150 "What is *identifier*?" questions for techniques, groups, software and CVEs, with name-phrased twins for the ATT&CK kinds; names shared by two documents are dropped. `hybrid_eval.py` runs all three retrievers twice each, requires each to agree with itself exactly, and writes every ranked list to `reports/data/hybrid_retrieval.json`. The probes are reported separately from the gold set, because they favour a lexical retriever by construction.

### 10.7 Findings reader (`eval/findings.py`)

Reads the committed JSON and JSONL dumps into the structure the findings page renders. Attack outcomes are reported as low–high ranges, with an `unstable` list naming the attacks that moved; they are never averaged, since an averaged 0.67 would present a coin flip as a measurement. Missing evidence files are reported in a `sources` field, not silently skipped.

---

## 11. API and web interface

### 11.1 Endpoints (`api/main.py`)

| Method and path | Purpose |
|---|---|
| `GET /health` | Process liveness |
| `GET /api/health/services` | Reachability of Qdrant and Ollama, reported separately from liveness |
| `GET /api/retrievers` | `dense`, `hybrid` and `hybrid_text`, with the collection each uses |
| `GET /api/defenses` | Defence names in registry order |
| `GET /api/principals` | Demo identities and their labels |
| `POST /api/search` | Retrieval only |
| `POST /api/ask` | Full answer; optional `defenses` list and `retrieval` name |
| `GET /api/attacks` | Attack ids, families, descriptions and target questions; **never the document text** |
| `POST /api/attacks/run` | Runs one committed attack by `attack_id`, never posted text, under a chosen defence set and retriever |
| `GET /api/findings` | The committed evidence, as read by `eval/findings.py` |
| `GET /` | Console page |
| `GET /findings` | Findings page |

### 11.2 Request handling

- **Principal.** The principal is named in the `X-Principal-Id` header. The server looks up that identity's clearance in its own configuration; the caller cannot set a clearance, and an unknown id returns HTTP 400. This is a laboratory affordance, labelled as such in the configuration, the interface and the reports.
- **Per-request defences and retriever.** The embedder, the generator and one store per (retriever, segregated) pair are built once at startup and held in a service container. The answer pipeline is composed per request by `factory.compose_answer_pipeline`. `defenses: null` means the server's configured set, while `[]` means explicitly undefended; collapsing the two would make the comparison the console exists for inexpressible. Retrievers are loaded from the committed overlays, so the interface cannot drift from the measured configurations.
- **Attack runner.** One run at a time, under a lock, with HTTP 409 if busy, because a run indexes into the live collection and two overlapping runs would each see the other's document. The attack document is removed in `finally`, and the collection count is checked afterwards.
- **A refused feature.** A count of passages withheld from the caller was rejected. It would require a second retrieval at a clearance the caller does not hold. Each passage instead shows its TLP level and trust tier.

### 11.3 Pages

- **Console (`index.html`).** Hand-written HTML and JavaScript with no framework. It offers a question box, principal and retriever selectors, and one toggle per defence, and shows the answer rendered as a Markdown subset over escaped HTML, *including images and links*. That rendering surface is what the exfiltration experiment measures; frameworks such as Streamlit or Gradio would sanitise it away. It also shows the retrieved passages with TLP and trust badges, and a defence panel distinguishing *ran*, *abstained (could not judge)* and *not enabled*, together with a service-health strip.
- **Findings (`findings.html`).** Renders the committed evidence and contacts no service, so it works with Docker and Ollama stopped. The utility charts are faceted per model and protocol, each with its own scale, because a shared axis compressed the within-model comparison into a few pixels. Chart colours were checked with a colour-difference validator; an initial grey-versus-accent pair failed the normal-vision threshold and was replaced.

### 11.4 Interface contract tests

`tests/test_ui_contract.py` parses the endpoint paths out of both HTML pages and checks that each exists on the app, and lists the response fields the pages read and checks that each is returned. Two assertions check meaning, not shape: the Markdown image rendering path must remain in the console, and an abstention must continue to read as "could not judge". Both halves were mutation-checked, in the sense that each was confirmed to fail when the thing it guards was broken.

---

## 12. Quality engineering

### 12.1 Tests

There are 415 tests in `tests/`, one module per component. They need no running service: Qdrant and Ollama are replaced by in-memory fakes that implement the ports.

| Area | Test modules |
|---|---|
| Domain and config | `test_config`, `test_access_control` |
| Sources | `test_attack_cti`, `test_nvd_api`, `test_nvd_cve`, `test_vendor_report`, `test_internal_notes` |
| Chunking and ingest | `test_chunking`, `test_pipeline` |
| Index | `test_qdrant_store`, `test_segregated_store`, `test_bm25`, `test_hybrid_retrieval`, `test_mean_pooled_embedder` |
| Generation | `test_generator`, `test_generator_bounds` |
| Attacks | `test_attacks`, `test_exfiltration_sink` |
| Defences | `test_defense_registry`, and one module per defence |
| Inversion | `test_inversion_sample`, `test_inversion_backfill`, `test_inversion_export`, `test_inversion_score`, `test_inversion_reidentify` |
| Evaluation | `test_metrics`, `test_answer_metrics`, `test_benchmark`, `test_paired`, `test_probes`, `test_findings` |
| API | `test_api`, `test_ui_contract` |

Several tests encode decisions rather than behaviour, so that a future edit cannot quietly reverse them:

- the NVD terms-of-use obligations;
- the injection screen accepting an UNTRUSTED benign document and rejecting an AUTHORITATIVE instructing one;
- corroboration not firing on an unrelated authoritative passage;
- the segregated merge reproducing the single-collection ranking;
- an unknown defence name raising an error.

### 12.2 Static analysis

`ruff check` and `ruff format --check` run over `src`, `tests` and `scripts`. `mypy` runs in strict mode over `src` and `scripts`. The tests are not type-checked in strict mode; doing so reports 53 pre-existing errors, which are out of scope. The Makefile's `lint` target runs all three.

### 12.3 Continuous integration

`.github/workflows/ci.yml` runs on pushes to `main` and on pull requests, using Ubuntu and Python 3.11. It installs CPU-only PyTorch from the PyTorch CPU index (the GPU wheels are about 2.5 GB, and CI never runs a model), installs the package with the `dev` and `pdf` extras, then lints, type-checks and runs the tests with coverage.

### 12.4 Repository hygiene

`.gitignore` excludes data, models, PDFs, virtual environments, caches, `.env`, rendered plots and sweep console logs. Its `/data/` rule is anchored to the repository root; unanchored, `data/` also matched `reports/data/`, and evidence files were silently left out of commits. Metric dumps are committed as evidence, and plots are not, because they are regenerable.

---

## 13. Defects found and fixed

Defects are grouped by the milestone in which they were found. Most were invisible to unit tests and surfaced only against real data.

### 13.1 M1: ATT&CK ingestion and retrieval

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | Recall@5 = 0.034 (chance) | Group-to-technique evidence exists only as STIX `uses` relationships; 5 of 941 gold pairs had evidence in any indexed text | Fold the 17,136 procedure-example descriptions into technique documents; coverage 941/941 |
| 2 | Technique documents lacked detection sections | ATT&CK v18 moved detection to `x-mitre-detection-strategy` and `x-mitre-analytic` objects | Join them back through `detects` relationships and analytic references |
| 3 | Stack overflow on 569 of 1,757 documents (every mitigation) | The recursive splitter's overlap carry could emit a final piece identical to its input, and recursion restarted from the coarsest separator | Thread a separator index so recursion always continues with a finer separator; termination is structural |
| 4 | Two candidate bundles of different sizes | The older `mitre/cti` mirror lags the canonical repository | Use `mitre-attack/attack-stix-data` and fix it before recording any baseline |

### 13.2 M2: generation and access control

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 5 | A chunker change would leave both strategies in one collection | Content-addressed ids mean a different chunking writes new points rather than replacing | `ingest --reset` deletes a source's points before re-ingesting |
| 6 | Reset-by-name could delete the wrong corpus | Source `name` and `source_type` coincided for ATT&CK by accident | Sources declare their own `source_type`, and reset deletes by it |
| 7 | First generate call would fail | `qwen2.5:7b-instruct` is not a tag Ollama publishes | Use `qwen2.5:7b` |
| 8 | Every citation ended in a replacement character on Windows | The console defaults to cp1252, and citations contain an em dash | `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` in the CLI |
| 9 | Access control could not be demonstrated | Every indexed document was TLP:CLEAR | Author the 16 classified internal notes |

### 13.3 M3: CVE and vendor corpora

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 10 | No CVE links found in ATT&CK | ATT&CK has no structured CVE field; mentions are prose | Regular expression over indexed `description` fields only |
| 11 | Projected 26,005 CVE chunks, many consisting only of URLs | Reference URL lists were about 70% of each CVE document | Index reference *meaning* (tags and count), keeping the canonical NVD URL |
| 12 | CVSS missing for slices of CVEs | NVD holds several CVSS versions and not every CVE has each | Newest version first; NVD Primary over vendor Secondary |
| 13 | Rejected CVEs and missing records | NVD tombstones; ATT&CK cites a CVE NVD never held | Skip tombstones; treat an empty result as data |
| 14 | Corpus would change between runs | A window anchored to "now" | Fixed `window_end: 2026-09-01` |
| 15 | A vendor report extracted zero characters | The PDF is entirely rasterised | Extraction floor of 200 characters per page rejects it; the next year's edition is used; OCR rejected as trading a loud failure for silent transcription errors |
| 16 | Doubled headings, dot leaders, running footers in extracted text | Faux-bold rendering, contents pages, footers whose page number varies | Per-line pair collapse, leader removal, frequency-based furniture detection after normalising digits |
| 17 | HTTP 403 from one vendor CDN | It rejects httpx's default User-Agent | Send a descriptive project User-Agent; no browser impersonation |
| 18 | Identifier queries matched the "shape" of CVE records (0.014 score margin) | Templated CVE documents cluster in embedding space | Recorded; addressed later by hybrid retrieval |

### 13.4 M4: attack harness

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 19 | Logs suggested the attack document held all five top-k slots | The runner printed `source_ref`, which an attack may share with a genuine document | Record and report retrieved passages by `doc_id` and `source_type` (corrected in M6) |
| 20 | An attack could never be retrieved for its identifier query | Dense MiniLM cannot match identifiers | Retarget that attack to a technique question; recorded as a retriever weakness |

### 13.5 M5: inversion

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 21 | The documented path for adding GTR vectors would have erased the MiniLM index | Qdrant upsert replaces the point, dropping other named vectors | `attach_vectors` via `update_vectors`; the CLI refuses a destructive ingest and warns when a re-ingest drops secondary vectors |
| 22 | Reconstructions would have been fluent nonsense | sentence-transformers' GTR pipeline adds a Dense projection and L2 normalisation; the corrector expects a raw masked mean pool. Cosine between the two encodings of one sentence: 0.018 | `MeanPooledEncoderEmbedder`, verified at maximum absolute difference 0.0 against vec2text's code |
| 23 | Input out of the corrector's distribution | 512-character chunks against a corrector trained on 32 tokens | A three-bundle design separating length from magnitude |
| 24 | Stored vectors had norm exactly 1.000 | Qdrant normalises on write under cosine distance (verified with a norm-5 probe) | Kept as a real property of the deployment; the unnormalised bundle isolates it |
| 25 | 26 of 334 bundle-versus-index cosines below 0.999 | A tokenise-decode round trip in sentencepiece is lossy; near-zero-mean vectors made the error visible | Encode the original text, not the decoded tokens; cosines 1.000000 afterwards |
| 26 | `vec2text` install failed | `rouge-score` is sdist-only and builds an invalid wheel under modern build isolation | Install it first with `--no-build-isolation` |
| 27 | `import vec2text` failed on Windows | It imports the Unix-only `resource` module (used only in a training entry point) | Stub `resource` in `sys.modules` before import |
| 28 | Corrector load failed with a meta-device error | transformers 5 forbids the nested `from_pretrained` vec2text performs | Pin the 2024-era stack in `.venv-inv` only |
| 29 | CUDA out of memory | Batch 16 or 32 × beam 4 means 64 to 128 concurrent sequences on 8 GB | Batch 4, which is also the fastest (0.13 vectors/s) |
| 30 | Re-identification reported 0/54 disclosures while working | The metric required the neighbour's identifier to appear verbatim; analysts write prose | Content-word subject metric; the strict column kept alongside |
| 31 | Stolen vendor and deep ATT&CK chunks matched nothing | The attacker's reference embedded whole documents, and MiniLM reads 256 tokens | `--passage-chars`; width chosen by tokenising every passage (1,000 characters truncated 60% of ATT&CK passages; 600 truncated 0%) |

### 13.6 M6: defences

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 32 | Candidate screen patterns fired on the real corpus | Patterns reasoned about, not measured | Measure every pattern on all 40,815 passages; drop the two that fired on Scheduled Task and a malware module name |
| 33 | A defence keyed on trust tier would score 100% at zero cost | Every attack document is UNTRUSTED and nothing else is | Rule: defences act on attacker-controlled properties only; tests pin both directions |
| 34 | D6 refused 0% of legitimate questions | Only 4 passages sit at or below its default threshold | Add a VENDOR-threshold overlay to measure the cost (33%) |

### 13.7 M7: benchmark

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 35 | The same configuration scored 2/7 then 3/7 | The sink bound an ephemeral port; the port number appeared in the attack document text, changing its chunk id, embedding and rank | Fixed port 24601, fatal on bind failure (a silent fallback would reintroduce the variance) |
| 36 | One request exceeded a 600 s timeout | `num_predict` was unset; output was unbounded while input was bounded | `num_predict: 1024`; the earlier 39-cell run was discarded |
| 37 | The M6 D2 retrieval row did not reproduce | Most likely an unrecorded index state at M6 time (the config and code had no diff; approximate search matched exact search) | Corrected in the reports; both arms of every comparison are now measured in the same session |
| 38 | Measuring D5 against its own store would report it as worthless | The segregated store is the defender's view, covering both collections | `attacker_public_view.yaml`: a plain store over the public collection only |
| 39 | D6 passed silently on 1.5b | The model rarely cites; the rule had nothing to inspect | Explicit abstention recorded on the answer, in results and in the console |
| 40 | Ollama timeouts after about 75 minutes of load | Sustained load on the local server | Resumable per-cell JSONL output |
| 41 | A sweep ran 21 minutes on CPU | The dedicated GPU was powered off by the laptop's power mode; Ollama detects GPUs only at startup | A load guard warns when `size_vram < size`; Ollama is restarted after the GPU returns |
| 42 | A false "fully reproducible" result during investigation | Force-killing `ollama.exe` orphaned `llama-server` processes that held VRAM | Stop both processes together and check `nvidia-smi` before any determinism measurement |

### 13.8 Temperature-0 nondeterminism

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 43 | Up to 72 of 200 answers differed between identical runs | A first generation depends on the cache state left by the previous prompt | `--warm`: generate twice and keep the second (57/200 → 1/200 within a process) |
| 44 | Warmed cells still differed across processes (13–31/200) | The fix holds within one process only | Positional control: compare a treated cell with an untreated cell at the same position; the help text was corrected |

### 13.9 Hybrid retrieval

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 45 | 468 of 1,250 queries came back in a different order on an immediate repeat | RRF scores tie by construction, and Qdrant returns tied points in no stable order; the cut to k was made on the server | Fetch all fused candidates, sort by score then chunk id, and cut in the adapter; all arms repeat exactly |
| 46 | Lexical matching would fail on identifiers | Generic tokenisation splits `T1055.001` and `CVE-2024-3094` | Identifier-first tokeniser |
| 47 | Lexical scores would be zero in a new process | Python `hash()` is salted per process | Stable digest-based term ids |
| 48 | The lexical ranking could return passages above the caller's clearance | Filter placed on the dense prefetch only | Filter on both prefetches |
| 49 | Hybrid plus D5 would silently void D5's retrieval-equivalence result | Score-based merging of RRF scores | Refused at config validation |

### 13.10 Interface

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 50 | The demo application ran completely undefended | The API built one pipeline at startup from `base.yaml`, where `defenses: []` | Per-request defence sets; a composable pipeline factory |
| 51 | Identifier questions returned nothing relevant in the console | The server used dense retrieval only | Per-request retriever selector loaded from the committed overlays |
| 52 | Pages could break silently when the API changed | No test linked the HTML to the routes | Contract tests parsing the pages, mutation-checked |
| 53 | Evidence dumps missing from commits | Unanchored `data/` ignore rule matched `reports/data/` | Anchored `/data/` |

---

## 14. Measured performance

Measured on the hardware in Section 2.6.

| Operation | Time |
|---|---|
| MiniLM embedding, CPU | about 53 passages/s |
| Ingest `attack_cti` (20,751 chunks) | 7 min 19 s |
| Ingest `nvd_cve` (17,301 chunks) | 6 min 46 s |
| Ingest `vendor_report` (2,729 chunks) | about 3 min (two-thirds PDF parsing) |
| Full 40,815-chunk segregated build | about 17 min |
| CLI startup (imports and model load) | about 25 s |
| Hybrid collection build | about 40 s |
| NVD fetch with API key (5,170 CVEs, 176 requests) | about 2 min; roughly 40 min without a key |
| Answer latency in isolation | 7.0 s (7b), 4.3 s (1.5b) |
| Attack run, 7-attack corpus | about 114 s |
| Attack run, evasion corpus | about 43 s |
| M7 benchmark cell (attacks + 50 questions) | about 123 s |
| Utility-only cell, 200 questions, GPU | 5.9 min (7b), 3.5 min (1.5b); warmed 7b 10.4–11.1 min |
| `invert reidentify`, whole-document reference | about 4.5 min per arm |
| `invert reidentify`, 600-character passages (23,053 vectors) | 11–12 min per arm |
| vec2text, 334 vectors, batch 4 | about 43 min per bundle; 124 min for three |
| `qwen2.5:7b` resident VRAM with `num_ctx` 8192 | about 5.1 GB |

Downloads: ATT&CK bundle 54 MB; vendor PDFs 73 MB; `qwen2.5:7b` 4.7 GB; `qwen2.5:1.5b` 1 GB; MiniLM about 90 MB. For the optional inversion step: CUDA PyTorch 2.9 GB (about 9 GB installed), vec2text models about 3.3 GB.

---

## 15. Reproducibility controls

- **Fixed inputs:** the ATT&CK bundle source; the CVE window end date; vendor PDF hashes; seeds 1337 (gold set, benchmark) and 20260906 (inversion sample).
- **Fixed generation:** temperature 0, `num_ctx` 8192, `num_predict` 1024, and the second generation kept when `--warm` is used.
- **Named configurations:** every cell is a committed overlay; defence order is fixed in the registry; unknown names fail.
- **Pinned harness artefacts:** a fixed sink port; attack documents deleted in `finally`; the collection count verified.
- **Same-session controls:** the dense or undefended arm is re-measured alongside the treated arm, not quoted from a report.
- **Self-agreement checks:** every retrieval arm in the hybrid evaluation is run twice and must match exactly.
- **Positional controls** for answer-utility comparisons.
- **Stated index state:** every retrieval figure names the collection and point count it was measured on.
- **Committed evidence:** every reported number has a JSON or JSONL source in `reports/data/`.

---

## 16. Known limitations and open technical items

- **Hybrid retrieval with D5** needs rank-based fusion across two collections. It is not designed, and it is refused in configuration.
- **D2 under hybrid retrieval** lands one attack. Capping on the fused rank, or exempting the query's matched identifier, are untested ideas. Capping low-trust documents harder was rejected, because it reads the project's own labels.
- **ATT&CK campaign objects** (56 with ATT&CK ids) are not indexed as documents, though their procedure text is folded into techniques.
- **Multi-column PDF interleaving** is not fixed.
- **Chunk-overlap orphans.** Tiny tail chunks from the overlap carry remain. Fixing them would invalidate the M1 baseline.
- **The Docker API container** is defined but was not the path used for the published results. The measured path runs Qdrant in Docker and the API on the host.
- **Tests are not type-checked** in strict mode.
- **Unexplained phenomena:** the instability of exf-002 on 1.5b, and the mechanism of across-process variation at temperature 0.
- **The inversion environment is Windows-specific** in its workarounds. On Linux the `resource` stub is inactive, but the dependency pins still apply.

---

## Appendix A: command-line reference

All commands accept `--config/-c` (base YAML) and, where relevant, a repeatable `--overlay/-o`.

| Command | Purpose | Main options |
|---|---|---|
| `threatrag fetch [attack\|nvd\|vendor\|all]` | Download raw corpora to `data/raw/` | `--force` |
| `threatrag ingest` | Parse, chunk, embed and index enabled sources | `--source`, `--reset`, `--embedder` |
| `threatrag build-hybrid` | Copy a dense collection and add BM25 vectors | `--from`, `--reset` |
| `threatrag build-goldset` | Build the 200-question gold set | `--limit`, `--seed` |
| `threatrag query "<q>"` | Retrieval only | `--k`, `--clearance` |
| `threatrag ask "<q>"` | Full answer with citations | `--k`, `--clearance` |
| `threatrag eval-retrieval` | Retrieval metrics on the gold set | `--k` |
| `threatrag status` | Collection and service status | |
| `threatrag attack list` | List attacks in a directory | `--dir` |
| `threatrag attack run [id]` | Run one or all attacks | `--dir`, `--sink-port` |
| `threatrag attack clean` | Remove residual attack documents | |
| `threatrag invert prepare` | Sample, attach GTR vectors, export bundles | `--embedder`, `--size`, `--seed`, `--out`, `--control-tokens` |
| `threatrag invert score <file>` | Score reconstructions | `--embedder`, `--out` |
| `threatrag invert reidentify` | Re-identification attack | `--sources`, `--passage-chars`, `--size`, `--seed`, `--dump` |
| `threatrag benchmark` | The benchmark sweep | `--models`, `--questions`, `--sets`, `--repeats`, `--skip-attacks`, `--warm`, `--out`, `--fresh`, `--sink-port`, `--seed` |

Scripts:

| Script | Purpose |
|---|---|
| `scripts/vec2text_invert.py --in DIR --out DIR [--batch 4] [--steps 20] [--beam 4] [--limit N]` | Inversion, run in `.venv-inv` |
| `scripts/hybrid_eval.py` | Dense vs hybrid retrieval evaluation |
| `scripts/utility_report.py FILE` | Paired utility table over a sweep file |
| `scripts/answer_length.py` | Answer-length analysis |

---

## Appendix B: evidence files

| File | Content |
|---|---|
| `reports/data/m7_sweep.jsonl` | M7 benchmark cells: per-attack outcomes, retrieval, 50-question utility |
| `reports/data/utility_sweep.jsonl` | 200-question utility cells, every answer |
| `reports/data/utility_warm.jsonl` | Warmed 7b utility cells, including the positional control |
| `reports/data/repro_check.jsonl`, `repro_check_warm.jsonl` | Temperature-0 reproducibility checks |
| `reports/data/m7_answer_length.json` | Verbosity analysis |
| `reports/data/m5_inversion_stored.json`, `_unnormalized.json`, `_control.json` | Per-chunk reconstruction scores |
| `reports/data/m5_reidentify_rows.json` | M5 re-identification rows |
| `reports/data/m7_reidentify_undefended.json`, `m7_reidentify_public.json` | Index route, weaker attacker, both arms |
| `reports/data/reidentify_strong_undefended.json`, `reidentify_strong_public.json` | Index route, stronger attacker, both arms |
| `reports/data/hybrid_retrieval.json` | Every ranked list, per question, per retriever |
| `reports/data/hybrid_attacks.jsonl` and `hybrid_attacks/` | Dense vs hybrid attack matrix and per-run logs |
| `reports/data/hybrid_attacks_variants.jsonl` and `hybrid_attacks_variants/` | Three-retriever attack matrix and per-run logs |
