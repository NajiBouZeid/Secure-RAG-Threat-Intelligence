"""Command-line entry point.

Thin by design: every command resolves a config, builds objects through the
factory, and calls into the library. No logic lives here that the API or the
benchmark runner would need to duplicate.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from threatrag import factory
from threatrag.config import Config, load_config
from threatrag.domain.models import TLP, Document, Principal, SourceType
from threatrag.eval import goldset as goldset_module
from threatrag.eval.metrics import aggregate, score_query
from threatrag.ingest.pipeline import IngestStats
from threatrag.ingest.sources.nvd_api import NVD_ATTRIBUTION
from threatrag.security.attacks.loader import DEFAULT_ATTACK_DIR, load_attacks
from threatrag.security.attacks.sink import ExfiltrationSink
from threatrag.security.inversion.backfill import backfill_vectors
from threatrag.security.inversion.export import (
    TRUTH_FILE,
    export_control,
    export_targets,
    load_reconstructions,
    load_truth,
    load_vectors,
)
from threatrag.security.inversion.reidentify import build_reference, reidentify
from threatrag.security.inversion.sample import DEFAULT_SEED, sample_chunks
from threatrag.security.inversion.score import score_reconstructions

# Citations carry an em dash, and a Windows console defaults to cp1252, which
# renders it as a replacement character. errors="replace" keeps a legacy console
# printing something rather than raising mid-table.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(add_completion=False, help="Secure Threat Intelligence RAG toolkit.")
console = Console()

ConfigOption = Annotated[Path | None, typer.Option("--config", "-c", help="Base config YAML.")]
OverlayOption = Annotated[
    Path | None, typer.Option("--overlay", "-o", help="Experiment overlay merged over the base.")
]

GOLDSET_FILENAME = "attack_goldset.json"


def _config(config: Path | None, overlay: Path | None = None) -> Config:
    return load_config(config, overlay)


FETCHABLE = ("attack", "nvd", "vendor", "all")


def _fetch_attack(cfg: Config, force: bool) -> None:
    source = factory.build_attack_source(cfg)
    source.fetch(force=force)
    size_mb = source.path.stat().st_size / 1e6
    console.print(f"[green]OK[/] {source.path} ({size_mb:.1f} MB)")


def _fetch_nvd(cfg: Config, force: bool) -> None:
    source = factory.build_nvd_source(cfg)
    client = source.client

    # Says whether a key is in use, never what it is.
    console.print(client.describe())
    console.print(NVD_ATTRIBUTION, style="dim")
    if not client.has_key:
        console.print(
            "[yellow]No NVD_API_KEY set.[/] The fetch will run at the public rate "
            "limit; add a key to .env to go faster. It is resumable either way.",
        )

    with console.status("Fetching CVEs (cached responses cost no requests)..."):
        source.fetch(force=force)

    console.print(
        f"[green]OK[/] {source.selected} CVEs selected, {client.requests_made} requests made"
    )


def _fetch_vendor(cfg: Config, force: bool) -> None:
    source = factory.build_vendor_source(cfg)
    console.print(f"Fetching {len(source.specs)} vendor reports")
    source.fetch(force=force)

    table = Table(title="Vendor reports", show_header=True)
    table.add_column("id")
    table.add_column("publisher")
    table.add_column("via")
    table.add_column("MB", justify="right")
    table.add_column("sha256")
    for spec in source.specs:
        size = source.path_for(spec).stat().st_size / 1e6
        table.add_row(
            spec.id,
            spec.publisher,
            spec.via,
            f"{size:.1f}",
            source.observed.get(spec.id, "")[:12],
        )
    console.print(table)
    if any(spec.sha256 is None for spec in source.specs):
        console.print(
            "[yellow]Some reports have no sha256 in the manifest.[/] Paste the digests "
            "above into corpora/vendor_reports.yaml so a changed document is detected."
        )


@app.command()
def fetch(
    corpus: Annotated[
        str, typer.Argument(help=f"Corpus to download: {', '.join(FETCHABLE)}")
    ] = "attack",
    config: ConfigOption = None,
    force: Annotated[bool, typer.Option("--force", help="Re-download even if cached.")] = False,
) -> None:
    """Download raw corpora into the data directory."""
    cfg = _config(config)
    if corpus not in FETCHABLE:
        raise typer.BadParameter(f"Unknown corpus {corpus!r} (available: {', '.join(FETCHABLE)})")

    # ATT&CK first when fetching everything: the CVE selection reads its
    # cross-links out of the bundle, so it has to be on disk already.
    if corpus in ("attack", "all"):
        _fetch_attack(cfg, force)
    if corpus in ("nvd", "all"):
        _fetch_nvd(cfg, force)
    if corpus in ("vendor", "all"):
        _fetch_vendor(cfg, force)


@app.command()
def ingest(
    config: ConfigOption = None,
    overlay: OverlayOption = None,
    embedder: Annotated[str | None, typer.Option("--embedder", help="Embedding model key.")] = None,
    reset: Annotated[
        bool,
        typer.Option("--reset", help="Drop this corpus from the collection before indexing."),
    ] = False,
    source_filter: Annotated[
        str | None,
        typer.Option("--source", help="Ingest only this source; default is every enabled one."),
    ] = None,
) -> None:
    """Parse, chunk, embed and index the enabled corpora."""
    cfg = _config(config, overlay)
    sources = factory.build_sources(cfg)
    if source_filter is not None:
        available = [source.name for source in sources]
        sources = [source for source in sources if source.name == source_filter]
        if not sources:
            raise typer.BadParameter(f"No enabled source named {source_filter!r}; have {available}")
    if not sources:
        raise typer.BadParameter("No sources enabled in the config.")
    pipeline = factory.build_pipeline(cfg, embedder)
    store = factory.build_store(cfg)

    # Qdrant replaces a point on upsert, so ingesting a populated collection
    # under a second encoder writes points carrying only that encoder's vector
    # and silently drops the primary one -- every earlier retrieval number with
    # it. Adding a second encoder to an indexed corpus is `invert prepare`,
    # which attaches vectors instead of replacing points.
    if embedder is not None and embedder != cfg.embedding.primary and store.count() > 0:
        raise typer.BadParameter(
            f"Collection {cfg.vector_store.collection!r} already holds "
            f"{store.count()} points indexed with {cfg.embedding.primary!r}. Ingesting "
            f"it again with {embedder!r} would replace those points and drop the "
            f"{cfg.embedding.primary!r} vectors. Use `threatrag invert prepare` to attach "
            f"a second encoder's vectors, or point vector_store.collection somewhere else."
        )

    # The guard above catches writing the *wrong* encoder. This catches the
    # mirror case, which is the one that actually bit: re-ingesting a source
    # under the primary encoder replaces those points and drops any second
    # encoder's vectors attached to them by `invert prepare`. Nothing errors --
    # the ingest reports success and the inversion corpus is quietly gone. The
    # re-ingest is legitimate, so this warns and names the repair rather than
    # refusing.
    written = embedder or cfg.embedding.primary
    secondary = sorted(name for name in cfg.embedding.models if name != written)
    before = {name: store.count_with_vector(name) for name in secondary}
    at_risk = {name: count for name, count in before.items() if count > 0}
    if at_risk:
        for name, count in at_risk.items():
            console.print(
                f"[yellow]warning[/] {count} points carry {name!r} vectors. Re-ingesting "
                f"replaces the points it writes and drops that vector from them; "
                f"`threatrag invert prepare --embedder {name}` re-attaches it."
            )

    stats = IngestStats()
    for source in sources:
        # Chunk ids are content-addressed, so re-ingesting with a different
        # chunker writes new points rather than replacing the old ones: without
        # --reset the collection would quietly hold two strategies at once and
        # every metric measured against it would be meaningless.
        if reset:
            removed = store.delete_by_source_type(source.source_type.value)
            console.print(f"[yellow]reset[/] removed {removed} existing {source.name} chunks")

        console.print(f"Ingesting [bold]{source.name}[/] with chunker={cfg.chunking.strategy}")
        stats = stats.merge(pipeline.run(source.load()))

    table = Table(title="Ingestion", show_header=False)
    table.add_row("sources", ", ".join(source.name for source in sources))
    table.add_row("documents seen", str(stats.documents_seen))
    table.add_row("documents rejected", str(stats.documents_rejected))
    table.add_row("chunks indexed", str(stats.chunks_indexed))
    console.print(table)

    # Report the damage rather than leaving it to be discovered by a scoring
    # run that silently drops those chunks.
    for name, was in at_risk.items():
        now = store.count_with_vector(name)
        if now < was:
            console.print(
                f"[red]dropped[/] {was - now} {name!r} vectors ({now} left). Run "
                f"`threatrag invert prepare --embedder {name}` to re-attach them."
            )


@app.command(name="build-goldset")
def build_goldset(
    config: ConfigOption = None,
    limit: Annotated[int, typer.Option("--limit", help="Number of questions to keep.")] = 200,
    seed: Annotated[int, typer.Option("--seed", help="Sampling seed; keep fixed.")] = 1337,
) -> None:
    """Derive the retrieval gold set from ATT&CK group-to-technique relationships."""
    cfg = _config(config)
    source = factory.build_attack_source(cfg)
    gold = goldset_module.build_from_attack(source.group_technique_pairs(), limit=limit, seed=seed)
    path = gold.save(cfg.paths.eval_dir / GOLDSET_FILENAME)
    console.print(f"[green]OK[/] {len(gold.queries)} queries -> {path}")


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Question to retrieve for.")],
    config: ConfigOption = None,
    overlay: OverlayOption = None,
    k: Annotated[int | None, typer.Option("--k", help="Override top-k.")] = None,
    clearance: Annotated[
        TLP, typer.Option("--clearance", help="Caller TLP clearance.")
    ] = TLP.CLEAR,
) -> None:
    """Run one retrieval and print the ranked chunks."""
    cfg = _config(config, overlay)
    retriever = factory.build_retriever(cfg)
    principal = Principal(id="cli", clearance=clearance)
    results = retriever.retrieve(question, k=k, principal=principal)

    if not results:
        console.print("[yellow]No results.[/] Is the collection populated? Try `threatrag ingest`.")
        raise typer.Exit(code=1)

    table = Table(title=f"top-{len(results)} for {question!r}")
    table.add_column("score", justify="right")
    table.add_column("ref")
    table.add_column("title")
    table.add_column("tlp")
    for hit in results:
        table.add_row(
            f"{hit.score:.3f}", hit.chunk.source_ref, hit.chunk.title, hit.chunk.tlp.value
        )
    console.print(table)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to answer.")],
    config: ConfigOption = None,
    overlay: OverlayOption = None,
    k: Annotated[int | None, typer.Option("--k", help="Override top-k.")] = None,
    clearance: Annotated[
        TLP, typer.Option("--clearance", help="Caller TLP clearance.")
    ] = TLP.CLEAR,
) -> None:
    """Answer a question over the indexed corpora, with citations."""
    cfg = _config(config, overlay)
    pipeline = factory.build_answer_pipeline(cfg)
    principal = Principal(id="cli", clearance=clearance)

    with console.status("Retrieving and generating..."):
        answer = pipeline.answer(question, principal=principal, k=k)

    console.print(Panel(answer.text, title=f"{answer.model} @ clearance={clearance.value}"))

    if answer.citations:
        table = Table(title="Citations", show_header=False)
        for index, citation in enumerate(answer.citations, start=1):
            table.add_row(str(index), citation)
        console.print(table)

    # Surfaced, not suppressed: an invented marker is the signal that the answer
    # drifted off its evidence, and hiding it here would hide it from M7 too.
    if answer.unsupported_citations:
        console.print(
            f"[yellow]unsupported citations[/]: {', '.join(answer.unsupported_citations)}"
        )


@app.command(name="eval-retrieval")
def eval_retrieval(
    config: ConfigOption = None,
    overlay: OverlayOption = None,
    k: Annotated[int | None, typer.Option("--k", help="Override top-k.")] = None,
) -> None:
    """Score retrieval against the ATT&CK gold set."""
    cfg = _config(config, overlay)
    gold = goldset_module.GoldSet.load(cfg.paths.eval_dir / GOLDSET_FILENAME)
    retriever = factory.build_retriever(cfg)
    top_k = k or cfg.retrieval.top_k

    # Chunks are scored by their document's ATT&CK id, so several chunks of one
    # technique count as one hit rather than inflating recall.
    scores = []
    with console.status("Scoring retrieval..."):
        for item in gold.queries:
            hits = retriever.retrieve(item.question, k=top_k)
            seen: list[str] = []
            for hit in hits:
                if hit.chunk.source_ref not in seen:
                    seen.append(hit.chunk.source_ref)
            scores.append(score_query(seen, item.relevant, top_k))

    summary = aggregate(scores)
    table = Table(title=f"Retrieval quality (k={top_k}, chunker={cfg.chunking.strategy})")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for name, value in summary.as_row().items():
        table.add_row(name, str(value))
    console.print(table)


@app.command()
def status(config: ConfigOption = None) -> None:
    """Show what is indexed."""
    cfg = _config(config)
    store = factory.build_store(cfg)
    console.print(f"collection [bold]{cfg.vector_store.collection}[/]: {store.count()} chunks")


attack_app = typer.Typer(
    add_completion=False, help="Run the M4 adversarial corpus against the pipeline."
)
app.add_typer(attack_app, name="attack")


@attack_app.command("list")
def attack_list(
    directory: Annotated[
        Path, typer.Option("--dir", help="Attack corpus directory.")
    ] = DEFAULT_ATTACK_DIR,
) -> None:
    """List the attacks in the corpus without running them."""
    attacks = load_attacks(directory)
    table = Table(title=f"{len(attacks)} attacks in {directory}")
    table.add_column("id")
    table.add_column("family")
    table.add_column("target query")
    for attack in attacks:
        table.add_row(attack.id, attack.family.value, attack.target_query)
    console.print(table)


@attack_app.command("run")
def attack_run(
    attack_id: Annotated[
        str | None, typer.Argument(help="Run one attack by id; default is all.")
    ] = None,
    config: ConfigOption = None,
    overlay: OverlayOption = None,
    directory: Annotated[
        Path, typer.Option("--dir", help="Attack corpus directory.")
    ] = DEFAULT_ATTACK_DIR,
) -> None:
    """Run attacks against the live index and report which landed.

    Needs Qdrant and the generator running: each attack indexes a poison
    document, runs its query, and removes the poison again.
    """
    cfg = _config(config, overlay)
    attacks = load_attacks(directory)
    if attack_id is not None:
        attacks = [a for a in attacks if a.id == attack_id]
        if not attacks:
            raise typer.BadParameter(f"No attack with id {attack_id!r}")

    # The sink is loopback-only and inert unless an exfiltration beacon fires at
    # it; it is held open for the whole batch so its port is stable across runs.
    with ExfiltrationSink() as sink:
        runner = factory.build_attack_runner(cfg, sink=sink)
        console.print(f"exfiltration sink listening on [dim]{sink.base_url}[/]")

        results = []
        with console.status("Running attacks..."):
            for attack in attacks:
                results.append(runner.run(attack))

    table = Table(title="Attack results")
    table.add_column("id")
    table.add_column("family")
    table.add_column("result")
    for result in results:
        verdict = "[red]LANDED[/]" if result.succeeded else "[green]blocked[/]"
        table.add_row(result.attack_id, result.family.value, verdict)
    console.print(table)

    for result in results:
        console.print(f"\n[bold]{result.attack_id}[/] — {result.target_query}")
        for criterion in result.criteria:
            mark = "[red]x[/]" if criterion.passed else "[green].[/]"
            console.print(f"  {mark} {criterion.detail}")

    landed = sum(1 for r in results if r.succeeded)
    console.print(f"\n[bold]{landed}/{len(results)}[/] attacks landed against the baseline.")


@attack_app.command("clean")
def attack_clean(config: ConfigOption = None) -> None:
    """Remove any leftover attack documents from the collection.

    A safety net: the runner already deletes its poison in a finally block, so
    this only matters after a hard-killed run.
    """
    cfg = _config(config)
    store = factory.build_store(cfg)
    removed = store.delete_by_source_type(SourceType.SYNTHETIC_ADVERSARIAL.value)
    console.print(f"[yellow]removed[/] {removed} synthetic-adversarial chunks")


invert_app = typer.Typer(
    add_completion=False, help="Run the M5 embedding-inversion attack against the index."
)
app.add_typer(invert_app, name="invert")

DEFAULT_INVERSION_DIR = Path("data/inversion")

# The control bundle lives beside the main one so `invert score --out` can
# point at either without a second set of flags.
# Three bundles, because a weak reconstruction has three possible causes and
# they mean different things. The stored one is what a thief of this index
# actually holds; the other two remove one confound each.
CONTROL_SUBDIR = "control"
UNNORMALIZED_SUBDIR = "unnormalized"

# ATT&CK and NVD are published, so an attacker can rebuild them himself. Vendor
# PDFs are omitted by default only because they are gitignored and may not be
# on disk, not because they are secret.
PUBLIC_SOURCES = (SourceType.ATTACK_CTI.value, SourceType.NVD_CVE.value)


@invert_app.command("prepare")
def invert_prepare(
    config: ConfigOption = None,
    overlay: OverlayOption = None,
    embedder: Annotated[
        str, typer.Option("--embedder", help="Encoder to attack; needs a public corrector.")
    ] = "gtr-base",
    size: Annotated[int, typer.Option("--size", help="Sampled chunks outside the census.")] = 300,
    seed: Annotated[int, typer.Option("--seed", help="Sampling seed; record it with results.")] = (
        DEFAULT_SEED
    ),
    out_dir: Annotated[
        Path, typer.Option("--out", help="Where to write the bundles.")
    ] = DEFAULT_INVERSION_DIR,
    control_tokens: Annotated[
        int,
        typer.Option(
            "--control-tokens",
            help="Also export a bundle at the corrector's training length; 0 to skip.",
        ),
    ] = 32,
) -> None:
    """Sample chunks, attach the attacked encoder's vectors, and export the bundle.

    Idempotent: re-running with the same seed selects the same chunks and
    overwrites their vectors with equal ones. The GTR vectors are attached to
    the existing points, so the MiniLM index the rest of the project measures
    against is untouched.
    """
    cfg = _config(config, overlay)
    store = factory.build_store(cfg)
    if store.count() == 0:
        raise typer.BadParameter(
            f"Collection {cfg.vector_store.collection!r} is empty; run `threatrag ingest` first."
        )

    sample = sample_chunks(store, size=size, seed=seed)
    console.print(
        f"sampled [bold]{sample.size}[/] chunks (seed {sample.seed}) from "
        f"{sum(sample.population.values())}: "
        + ", ".join(f"{name} {count}" for name, count in sample.selected.items())
    )

    encoder = factory.build_embedder(cfg, embedder)
    with console.status(f"Embedding {sample.size} chunks with {embedder}..."):
        stats = backfill_vectors(store, encoder, sample.chunks)
    console.print(f"attached [bold]{stats.vectors_attached}[/] {embedder} vectors")
    if stats.skipped:
        console.print(f"[yellow]{stats.skipped}[/] sampled chunks had no point in the index")

    manifest = export_targets(store, sample, vector_name=embedder, out_dir=out_dir)

    table = Table(title="Inversion bundle")
    table.add_row("vectors", f"{manifest.exported} x {manifest.dim}d")
    table.add_row("upload", ", ".join(manifest.upload_files))
    table.add_row("answer key", f"{TRUTH_FILE} (stays local)")
    table.add_row("directory", str(out_dir))
    console.print(table)

    # Both extra bundles are embedded here rather than read from the store: the
    # collection has no spare named vector for another convention, Qdrant cannot
    # add one to an existing collection, and anything written to a cosine
    # collection comes back normalised anyway, which is the confound one of them
    # exists to remove.
    if control_tokens > 0:
        control_dir = out_dir / CONTROL_SUBDIR
        control_encoder = factory.build_control_embedder(cfg, embedder, control_tokens)
        with console.status(f"Embedding the {control_tokens}-token control bundle..."):
            control = export_control(
                sample, control_encoder, out_dir=control_dir, variant="len32_unnormalized"
            )
        console.print(
            f"control bundle: [bold]{control.exported}[/] vectors at "
            f"{control.max_tokens} tokens in {control_dir} — the corrector's own "
            f"training length"
        )

        raw_dir = out_dir / UNNORMALIZED_SUBDIR
        raw_encoder = factory.build_control_embedder(cfg, embedder, None)
        with console.status("Embedding the full-length unnormalised bundle..."):
            raw = export_control(sample, raw_encoder, out_dir=raw_dir, variant="full_unnormalized")
        console.print(
            f"unnormalised bundle: [bold]{raw.exported}[/] full-length vectors in "
            f"{raw_dir} — separates lost magnitude from chunk length as the cause "
            f"of any failure on the stored bundle"
        )
    console.print(
        Panel(
            f"Upload only [bold]{'[/] and [bold]'.join(manifest.upload_files)}[/] to the GPU "
            f"runner. {TRUTH_FILE} is the answer key: sending it would make the result "
            f"circular, and it carries the AMBER and RED note text.",
            title="what leaves the machine",
        )
    )


@invert_app.command("score")
def invert_score(
    reconstructions: Annotated[
        Path, typer.Argument(help="JSONL of chunk_id + reconstruction from the inversion run.")
    ],
    config: ConfigOption = None,
    overlay: OverlayOption = None,
    embedder: Annotated[
        str | None,
        typer.Option("--embedder", help="Re-embed reconstructions for round-trip cosine."),
    ] = None,
    out_dir: Annotated[
        Path, typer.Option("--out", help="Directory holding the exported bundle.")
    ] = DEFAULT_INVERSION_DIR,
) -> None:
    """Score reconstructions against the local answer key."""
    truth = load_truth(out_dir / TRUTH_FILE)
    recovered = load_reconstructions(reconstructions)
    if not recovered:
        raise typer.BadParameter(f"No reconstructions read from {reconstructions}")

    stolen = None
    encoder = None
    if embedder is not None:
        cfg = _config(config, overlay)
        encoder = factory.build_embedder(cfg, embedder)
        stolen = load_vectors(out_dir)

    scores = score_reconstructions(truth, recovered, stolen_vectors=stolen, embedder=encoder)

    table = Table(title=f"Inversion, {scores.scored} targets scored")
    table.add_column("group")
    table.add_column("n", justify="right")
    table.add_column("exact", justify="right")
    table.add_column("token F1", justify="right")
    table.add_column("cosine", justify="right")
    table.add_column("secrets", justify="right")
    table.add_column("fully leaked", justify="right")
    for group in [scores.overall, *scores.by_source_type, *scores.by_tlp]:
        cosine = f"{group.mean_cosine:.3f}" if group.mean_cosine is not None else "-"
        table.add_row(
            group.name,
            str(group.count),
            f"{group.exact_match_rate:.3f}",
            f"{group.mean_token_f1:.3f}",
            cosine,
            f"{group.secret_recovery_rate:.3f}",
            str(group.fully_leaked),
        )
    console.print(table)
    console.print(f"corpus BLEU-4: [bold]{scores.corpus_bleu:.4f}[/]")
    if scores.unmatched:
        console.print(f"[yellow]{len(scores.unmatched)}[/] reconstructions had no answer-key entry")


@invert_app.command("reidentify")
def invert_reidentify(
    config: ConfigOption = None,
    overlay: OverlayOption = None,
    embedder: Annotated[
        str | None, typer.Option("--embedder", help="Encoder whose vectors were stolen.")
    ] = None,
    size: Annotated[int, typer.Option("--size", help="Sampled chunks outside the census.")] = 300,
    seed: Annotated[int, typer.Option("--seed", help="Sampling seed.")] = DEFAULT_SEED,
    sources: Annotated[
        str, typer.Option("--sources", help="Public corpora the attacker rebuilds.")
    ] = ",".join(PUBLIC_SOURCES),
    dump: Annotated[
        Path | None,
        typer.Option("--dump", help="Write the per-chunk rows as JSON evidence."),
    ] = None,
) -> None:
    """Match stolen vectors to public documents, no corrector required.

    The attack that still works on MiniLM. Uses the same seeded sample as
    `prepare`, so the two results describe the same chunks.
    """
    cfg = _config(config, overlay)
    store = factory.build_store(cfg)
    encoder = factory.build_embedder(cfg, embedder)

    sample = sample_chunks(store, size=size, seed=seed)
    stolen = store.get_vectors(encoder.name, [chunk.id for chunk in sample.chunks])
    if not stolen:
        raise typer.BadParameter(
            f"No {encoder.name!r} vectors stored for the sample; check the collection."
        )

    wanted = {name.strip() for name in sources.split(",") if name.strip()}
    public = [source for source in factory.build_sources(cfg) if source.source_type.value in wanted]
    if not public:
        raise typer.BadParameter(f"No enabled sources match {sorted(wanted)}")

    def documents() -> Iterator[Document]:
        for source in public:
            yield from source.load()

    with console.status("Rebuilding the public corpus as the attacker would..."):
        corpus, reference = build_reference(documents(), encoder)
    console.print(f"attacker reference: [bold]{corpus.size}[/] public documents")

    report = reidentify(sample.chunks, stolen, corpus, reference)

    table = Table(title="Re-identification from stolen vectors")
    table.add_column("population")
    table.add_column("n", justify="right")
    table.add_column("rate", justify="right")
    table.add_row(
        "public chunks recognised", str(report.recognisable), f"{report.top1_accuracy:.3f}"
    )
    table.add_row(
        "hidden chunks whose subject leaked",
        str(report.unrecognisable),
        f"{report.topic_disclosure_rate:.3f}",
    )
    console.print(table)

    # Broken out because the hidden population is not homogeneous: the internal
    # notes are genuinely absent from any public corpus, while vendor chunks are
    # only absent from the reference this run happened to build.
    corpora = Table(title="By corpus")
    corpora.add_column("source type")
    corpora.add_column("n", justify="right")
    corpora.add_column("recognised", justify="right")
    corpora.add_column("subject leaked", justify="right")
    corpora.add_column("id quoted", justify="right")
    for group in report.by_source_type:
        corpora.add_row(
            group.name,
            str(group.count),
            str(group.recognised),
            str(group.topic_hits),
            str(group.ref_hits),
        )
    console.print(corpora)
    console.print(
        f"top-1 hits {report.recognised}/{report.recognisable}, "
        f"subject leaked {report.topic_hits}/{report.unrecognisable}, "
        f"identifier quoted outright {report.ref_hits}/{report.unrecognisable}"
    )

    if dump is not None:
        # Per-chunk rows carry the nearest public document and its score but no
        # chunk text, so the evidence behind a published rate is checkable
        # without the note bodies leaving data/.
        dump.parent.mkdir(parents=True, exist_ok=True)
        dump.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            # Default translation writes CRLF on Windows, and git then warns on
            # every commit of a file that is meant to be a stable artefact.
            newline="\n",
        )
        console.print(f"wrote {len(report.rows)} rows to [bold]{dump}[/]")


if __name__ == "__main__":
    app()
