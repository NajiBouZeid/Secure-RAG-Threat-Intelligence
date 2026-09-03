"""Command-line entry point.

Thin by design: every command resolves a config, builds objects through the
factory, and calls into the library. No logic lives here that the API or the
benchmark runner would need to duplicate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from threatrag import factory
from threatrag.config import Config, load_config
from threatrag.domain.models import TLP, Principal
from threatrag.eval import goldset as goldset_module
from threatrag.eval.metrics import aggregate, score_query
from threatrag.ingest.pipeline import IngestStats

app = typer.Typer(add_completion=False, help="Secure Threat Intelligence RAG toolkit.")
console = Console()

ConfigOption = Annotated[Path | None, typer.Option("--config", "-c", help="Base config YAML.")]
OverlayOption = Annotated[
    Path | None, typer.Option("--overlay", "-o", help="Experiment overlay merged over the base.")
]

GOLDSET_FILENAME = "attack_goldset.json"


def _config(config: Path | None, overlay: Path | None = None) -> Config:
    return load_config(config, overlay)


@app.command()
def fetch(
    corpus: Annotated[str, typer.Argument(help="Corpus to download: attack")] = "attack",
    config: ConfigOption = None,
    force: Annotated[bool, typer.Option("--force", help="Re-download even if cached.")] = False,
) -> None:
    """Download raw corpora into the data directory."""
    cfg = _config(config)
    if corpus != "attack":
        raise typer.BadParameter(f"Unknown corpus {corpus!r} (available: attack)")

    source = factory.build_attack_source(cfg)
    source.fetch(force=force)
    size_mb = source.path.stat().st_size / 1e6
    console.print(f"[green]OK[/] {source.path} ({size_mb:.1f} MB)")


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


if __name__ == "__main__":
    app()
