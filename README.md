# Secure Threat Intelligence RAG

A retrieval-augmented threat intelligence assistant over MITRE ATT&CK, NVD CVE records and vendor threat reports, together with the attacks, defences and benchmark used to evaluate its security.

This README covers installation and reproduction only. For explanations and results, see:

- **[Project Report](docs/project_report.md)**: concepts, architecture, data, attacks, defences and results, written for readers new to the project.
- **[Technical Report](docs/technical_report.md)**: toolchain, repository structure, implementation, engineering defects and their fixes.
- **[Milestone reports](reports/)**: the original per-milestone write-ups (M1–M7 and hybrid retrieval), with the evidence they cite in `reports/data/`.

---

## Contents

1. [Requirements](#1-requirements)
2. [Installation](#2-installation)
3. [Services](#3-services)
4. [Build the corpus and index](#4-build-the-corpus-and-index)
5. [Use the assistant](#5-use-the-assistant)
6. [Reproduce the results](#6-reproduce-the-results)
7. [Tests and quality checks](#7-tests-and-quality-checks)
8. [Running in Docker](#8-running-in-docker)
9. [Troubleshooting](#9-troubleshooting)
10. [Repository layout](#10-repository-layout)
11. [Data sources, attribution and scope](#11-data-sources-attribution-and-scope)
12. [License](#12-license)

---

## 1. Requirements

### Software

| Component | Version | Notes |
|---|---|---|
| Python | 3.11 or later | |
| Docker and Docker Compose | recent | Runs Qdrant |
| Ollama | recent | Runs on the host, not in Docker |
| Git | any | |
| GNU Make | optional | Shortcuts only; every command is also given in full below |

### Hardware

| Task | Minimum | Used for the published results |
|---|---|---|
| Ingestion, retrieval, tests | Any modern CPU, 16 GB RAM | CPU embedding |
| Answer generation (`qwen2.5:7b`) | GPU with 6 GB or more VRAM recommended; CPU works but is slow | NVIDIA RTX 4060 Laptop, 8 GB |
| Embedding inversion (optional) | NVIDIA GPU with 8 GB VRAM | Same |

### Disk and download sizes

| Item | Download | Needed for |
|---|---|---|
| Python dependencies (PyTorch CPU included) | about 1 GB | Everything |
| MiniLM embedding model | about 90 MB | Everything |
| ATT&CK bundle | 54 MB | Index |
| NVD records | small (API responses) | Index |
| Vendor PDFs | 73 MB | Index |
| `qwen2.5:7b` | 4.7 GB | Answers, attacks, benchmark |
| `qwen2.5:1.5b` | 1 GB | Two-model benchmark |
| Qdrant storage (all collections) | about 1–2 GB on disk | Index |
| Inversion environment (optional) | about 3 GB PyTorch CUDA plus 3.3 GB models; about 12 GB on disk | Section 6.6 only |

### Time estimates (measured on the hardware above)

| Step | Time |
|---|---|
| Fetch all corpora with an NVD API key | about 3 min (about 40 min without a key) |
| Ingest all corpora (40,815 passages, CPU) | about 17 min |
| One attack run, seven attacks | about 2 min |
| Full benchmark (42 cells) | about 1.5 h |
| 200-question utility sweep | about 2 h unwarmed |
| Inversion, three bundles | about 2 h on GPU |

---

## 2. Installation

```bash
git clone https://github.com/NajiBouZeid/Secure-RAG-Threat-Intelligence.git
cd Secure-RAG-Threat-Intelligence

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev,pdf]"
```

or `make install` with the virtual environment active.

This installs the `threatrag` command. Every command below can also be run as `python -m threatrag.cli ...`.

### Environment file

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
|---|---|---|
| `NVD_API_KEY` | empty | Optional. Free from https://nvd.nist.gov/developers/request-an-api-key. Without it the NVD fetch is throttled to the public rate. Never put it in a config file. |
| `THREATRAG_QDRANT_URL` | `http://localhost:6333` | Qdrant address |
| `THREATRAG_OLLAMA_URL` | `http://localhost:11434` | Ollama address |
| `THREATRAG_CONFIG` | `configs/base.yaml` | Base configuration |
| `THREATRAG_DATA_DIR` | `data` | Raw data, gold set and inversion bundles |

---

## 3. Services

### Qdrant

```bash
docker compose up -d qdrant        # or: make up
```

Qdrant listens on `localhost:6333`. The image is pinned to `qdrant/qdrant:v1.19.0` to match the client. Data persists in the `qdrant_storage` Docker volume.

### Ollama

Install Ollama from https://ollama.com, then:

```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5:1.5b     # only needed for the two-model benchmark
ollama serve                 # if the server is not already running
```

The tag is `qwen2.5:7b`, not `qwen2.5:7b-instruct`.

Check that the model is fully on the GPU after a first request:

```bash
curl http://localhost:11434/api/ps
```

`size_vram` should equal `size`. Ollama detects GPUs only when it starts, so restart it if the GPU became available afterwards.

---

## 4. Build the corpus and index

### 4.1 Download

```bash
threatrag fetch all          # or: make fetch
```

This downloads ATT&CK first (the CVE selection reads it), then NVD, then the vendor PDFs, into `data/raw/`. Each vendor PDF is verified against the SHA-256 in `corpora/vendor_reports.yaml`, and a mismatch stops the fetch. Individual corpora can be fetched with `threatrag fetch attack`, `threatrag fetch nvd` or `threatrag fetch vendor`, and `--force` re-downloads.

The CVE selection uses a fixed end date (`2026-09-01`) and is reproducible. Because NVD records change over time, a later fetch may differ slightly in CVE descriptions or scores.

### 4.2 Ingest

```bash
threatrag ingest             # or: make ingest
threatrag status
```

Expected: `collection threatrag: 40815 chunks`.

| Source | Passages |
|---|---|
| `attack_cti` | 20,751 |
| `nvd_cve` | 17,301 |
| `vendor_report` | 2,729 |
| `internal_notes` | 34 |

To re-ingest one source after a change, use `threatrag ingest --source <name> --reset`. Without `--reset`, a change of chunker adds points instead of replacing them.

### 4.3 Gold set

```bash
threatrag build-goldset
```

This writes 200 questions to `data/eval/attack_goldset.json` (seed 1337).

---

## 5. Use the assistant

### Command line

```bash
threatrag query "What persistence techniques does APT29 use?"            # retrieval only
threatrag ask "How does Kerberoasting work?"                             # cited answer
threatrag ask "What was found in the treasury incident?" --clearance red # as a TLP:RED user
```

`--clearance` accepts `clear`, `green`, `amber` or `red`, and `--k` overrides top-k. Add `-o <overlay>` to use a defence set or retriever, for example:

```bash
threatrag ask "What is T1055.001?" -o configs/experiments/retrieval_hybrid.yaml
```

The hybrid overlays require the hybrid collections (Section 6.7).

### Web interface

```bash
uvicorn threatrag.api.main:app --port 8000          # or: make serve
```

| URL | Page |
|---|---|
| http://localhost:8000/ | Console: principal, retriever and defence selection, cited answers, audit panels, attack runner |
| http://localhost:8000/findings | Findings from the committed evidence; works with Qdrant and Ollama stopped |

The console's principal selector (analyst, senior analyst, IR lead) is a demo identity sent as an `X-Principal-Id` header. It is not authentication.

---

## 6. Reproduce the results

Every result is produced by a committed configuration overlay in `configs/experiments/`. Overlays are deep-merged over `configs/base.yaml` and can be combined by repeating `-o`. Unless stated otherwise, commands run against the `threatrag` collection built in Section 4, with `qwen2.5:7b`.

Expected values are those published in the reports. Retrieval figures should match to four decimals against an identical index. Attack and utility figures carry run-to-run variation, which is described in the reports; compare ranges, not single values.

### 6.1 Retrieval baseline (M1–M3)

```bash
threatrag eval-retrieval             # or: make eval-retrieval
```

Expected at k = 5: recall 0.1631, precision 0.1764, MRR 0.3128, nDCG 0.1823, hit rate 0.490.

**Chunking comparison (M1, ATT&CK only).** Each strategy is written to its own collection:

```bash
threatrag ingest --source attack_cti -o configs/experiments/chunking_structural.yaml
threatrag ingest --source attack_cti -o configs/experiments/chunking_fixed.yaml
threatrag ingest --source attack_cti -o configs/experiments/chunking_recursive.yaml
threatrag eval-retrieval -o configs/experiments/chunking_structural.yaml --k 10
threatrag eval-retrieval -o configs/experiments/chunking_fixed.yaml --k 10
threatrag eval-retrieval -o configs/experiments/chunking_recursive.yaml --k 10
```

Expected recall@10: structural 0.2311, fixed 0.2199, recursive 0.2629. See `reports/m1_baseline.md`.

### 6.2 Attacks (M4)

```bash
threatrag attack list
threatrag attack run                          # the seven-attack corpus, undefended
threatrag attack run --dir attacks/evasion    # the two evasion variants
```

Expected: 3 of 7 land undefended (see `reports/m4_attacks.md`). Each run indexes its attack document, evaluates, and removes the document. If a run is interrupted, run `threatrag attack clean`, then confirm with `threatrag status` that the collection is back to 40815.

Keep the default `--sink-port` (24601). An ephemeral port changes the attack documents and therefore retrieval.

### 6.3 Defences (M6)

One defence at a time:

```bash
threatrag attack run -o configs/experiments/defense_provenance_fence.yaml
threatrag attack run -o configs/experiments/defense_source_cap.yaml
threatrag attack run -o configs/experiments/defense_egress_filter.yaml
threatrag attack run -o configs/experiments/defense_injection_screen.yaml
threatrag attack run -o configs/experiments/defense_corroboration.yaml
threatrag attack run -o configs/experiments/defense_all.yaml
```

Repeat each with `--dir attacks/evasion` for the evasion corpus. The retrieval effect of the source cap:

```bash
threatrag eval-retrieval -o configs/experiments/defense_source_cap.yaml
```

Expected: recall@5 0.1665. `defense_corroboration_vendor.yaml` measures the corroboration defence at its lower threshold. See `reports/m6_defenses.md`.

**Corpus segregation** needs its own index, built with the overlay enabled (about 17 minutes):

```bash
threatrag ingest -o configs/experiments/defense_corpus_segregation.yaml
```

Expected: `threatrag_public` 40,787 points and `threatrag_restricted` 28 points.

### 6.4 Benchmark (M7)

```bash
threatrag benchmark --repeats 3
```

This covers seven defence sets × two models (`qwen2.5:7b`, `qwen2.5:1.5b`) × three repeats: 42 cells, about 1.5 hours. Results are appended per cell to `reports/data/m7_sweep.jsonl`. The run is resumable, since completed cells are skipped; `--fresh` starts over. To write elsewhere, use `--out <file>`.

The committed `reports/data/m7_sweep.jsonl` already contains the published cells. Use `--out` with a new file to reproduce them without mixing results. See `reports/m7_benchmark.md`.

### 6.5 Answer-utility study (200 questions)

```bash
threatrag benchmark --skip-attacks --questions 200 --out my_utility.jsonl
threatrag benchmark --skip-attacks --questions 200 --sets none --repeats 4 --out my_utility.jsonl
python scripts/utility_report.py my_utility.jsonl
```

Add `--warm` to generate each answer twice and keep the second. This takes about 75% longer and removes most within-process variation. On Windows, set `PYTHONIOENCODING=utf-8` before running `utility_report.py`.

### 6.6 Stolen-index attacks (M5)

**Re-identification** (CPU only):

```bash
# undefended index
threatrag invert reidentify --sources attack_cti,nvd_cve,vendor_report --passage-chars 600 \
    --dump my_reidentify_undefended.json
# stolen public collection, after building the segregated index (Section 6.3)
threatrag invert reidentify --sources attack_cti,nvd_cve,vendor_report --passage-chars 600 \
    -o configs/experiments/attacker_public_view.yaml --dump my_reidentify_public.json
```

Expected: 181/280 public passages recognised in both arms; note-subject disclosure 12/34 undefended and 2/6 on the public collection. About 12 minutes per arm.

**Reconstruction** (NVIDIA GPU, separate environment):

1. Prepare the bundles in the project environment. This samples 334 passages, attaches GTR vectors to them in Qdrant, and writes `data/inversion/`, `data/inversion/unnormalized/` and `data/inversion/control/`:

   ```bash
   threatrag invert prepare
   ```

2. Create the isolated inversion environment. vec2text must not be installed in `.venv`:

   ```bash
   python -m venv .venv-inv
   .venv-inv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
   .venv-inv/Scripts/python -m pip install setuptools wheel
   .venv-inv/Scripts/python -m pip install rouge-score --no-build-isolation
   .venv-inv/Scripts/python -m pip install vec2text
   .venv-inv/Scripts/python -m pip install "transformers==4.44.2" "huggingface_hub<1.0" \
       "tokenizers>=0.19,<0.20" "sentence-transformers==3.0.1" "datasets<3" "accelerate<1.0"
   ```

   On Linux, use `.venv-inv/bin/python`.

3. Invert each bundle. Keep `--steps 20 --beam 4` (the defaults) for comparable results; `--limit 5` runs a smoke test.

   ```bash
   .venv-inv/Scripts/python scripts/vec2text_invert.py --in data/inversion --out data/inversion --batch 4
   .venv-inv/Scripts/python scripts/vec2text_invert.py --in data/inversion/unnormalized --out data/inversion/unnormalized --batch 4
   .venv-inv/Scripts/python scripts/vec2text_invert.py --in data/inversion/control --out data/inversion/control --batch 4
   ```

4. Score each bundle in the project environment:

   ```bash
   threatrag invert score data/inversion/reconstructions.jsonl --out data/inversion
   threatrag invert score data/inversion/unnormalized/reconstructions.jsonl --out data/inversion/unnormalized
   threatrag invert score data/inversion/control/reconstructions.jsonl --out data/inversion/control
   ```

Expected exact match: stored 0.024, unnormalised 0.117, control 0.222. See `reports/m5_inversion.md`.

Re-ingesting a source after `invert prepare` drops the GTR vectors from the points it rewrites. The CLI warns when this happens; rerun `threatrag invert prepare` to re-attach them.

### 6.7 Hybrid retrieval

Build the two hybrid collections as copies of `threatrag`, which is left untouched as the dense control:

```bash
threatrag build-hybrid -o configs/experiments/retrieval_hybrid.yaml
threatrag build-hybrid -o configs/experiments/retrieval_hybrid_text.yaml
python scripts/hybrid_eval.py
```

Expected recall@5: dense 0.1631, hybrid 0.2079, hybrid (text only) 0.2153.

Attack comparison:

```bash
threatrag attack run
threatrag attack run -o configs/experiments/defense_all.yaml
threatrag attack run -o configs/experiments/retrieval_hybrid.yaml
threatrag attack run -o configs/experiments/retrieval_hybrid.yaml -o configs/experiments/defense_all.yaml
```

Repeat each with `--dir attacks/evasion`, and with `retrieval_hybrid_text.yaml`. Expected on the seven-attack corpus: dense 3/7 undefended and 0/7 defended; hybrid 5/7 and 1/7. See `reports/hybrid_retrieval.md`.

Hybrid retrieval cannot be combined with `defense_corpus_segregation.yaml`; the configuration refuses it.

---

## 7. Tests and quality checks

No running service is required.

```bash
pytest                              # 415 tests; or: make test
ruff check src tests scripts
ruff format --check src tests scripts
mypy src scripts                    # or run all three with: make lint
```

CI (`.github/workflows/ci.yml`) runs the same checks on every push to `main` and on every pull request.

---

## 8. Running in Docker

`docker-compose.yml` defines Qdrant and an API container. Ollama stays on the host, and the API reaches it via `host.docker.internal`.

```bash
docker compose up -d --build
```

The API is served on http://localhost:8000. `data/`, `configs/`, `corpora/` and `attacks/` are mounted from the host, so fetching and ingestion can be done from the host environment (Sections 4.1–4.2) against the same Qdrant instance. The published results were produced with the API running on the host (Section 5), not in the container.

---

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Connection refused` on port 6333 | Qdrant is not running. Start Docker, then `docker compose up -d qdrant`. |
| Generation fails or times out | Ollama is not running, or the model is not pulled. Run `ollama serve` and `ollama pull qwen2.5:7b`. |
| Generation is very slow | The model is on the CPU. Check `curl localhost:11434/api/ps` (`size_vram` should equal `size`) and restart Ollama after the GPU is available. The benchmark prints a warning in this case. |
| CUDA out of memory in Ollama after restarts | Orphaned `llama-server` processes hold VRAM. Stop `ollama` and `llama-server` together, then check `nvidia-smi`. |
| Collection count is not 40815 after an attack run | An interrupted run left its document behind. Run `threatrag attack clean`. |
| Retrieval numbers differ from the reports | Check `threatrag status` for the point count. Make sure no chunker change was ingested without `--reset`. |
| Garbled characters in the Windows console | Set `PYTHONIOENCODING=utf-8`. |
| Vendor fetch reports a hash mismatch | The URL now serves a different file. The manifest records the expected hash; do not update it without inspecting the new file. |
| NVD fetch is slow | No API key. Add `NVD_API_KEY` to `.env`. |
| `import vec2text` fails with `No module named 'resource'` | Use `scripts/vec2text_invert.py`, which stubs the module on Windows. Do not import vec2text directly. |
| vec2text fails with a meta-device error | transformers 5 is installed in `.venv-inv`. Apply the pins in Section 6.6. |

---

## 10. Repository layout

```
attacks/            Attack corpus (seven attacks; evasion/ holds two variants)
configs/            base.yaml and experiment overlays
corpora/            Internal notes (fictional) and the vendor PDF manifest
docs/               Project and technical reports
reports/            Milestone reports; data/ holds the committed evidence
scripts/            Inversion, hybrid evaluation and utility analysis scripts
src/threatrag/      The package (domain, ingest, index, rag, security, eval, api)
tests/              Test suite
```

A full description of every module is in the [Technical Report](docs/technical_report.md#3-repository-structure).

---

## 11. Data sources, attribution and scope

- **MITRE ATT&CK®** is a registered trademark of The MITRE Corporation. The bundle is downloaded from `mitre-attack/attack-stix-data`.
- **This product uses the NVD API but is not endorsed or certified by the NVD.** CVE descriptions are reproduced verbatim. Enumerated lists are bounded, and each truncation is marked in the text.
- **Vendor threat reports** remain the copyright of their publishers. The repository contains only a manifest of public URLs and hashes; the PDFs are downloaded at build time and are not redistributed.
- **The internal notes** in `corpora/internal_notes.yaml` are entirely fictional.
- **The attack files** in `attacks/` are inert YAML used to test a locally hosted, self-owned system. Nothing executes them automatically, exfiltration measurement uses only a listener on the local machine, and nothing in this repository targets a third-party service.

---

## 12. License

MIT. See [LICENSE](LICENSE).
