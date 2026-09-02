"""Shared helpers for corpus sources."""

from __future__ import annotations

from pathlib import Path

import httpx
from rich.progress import Progress, SpinnerColumn, TextColumn

DOWNLOAD_TIMEOUT = httpx.Timeout(30.0, read=300.0)


def download(url: str, destination: Path, *, force: bool = False) -> Path:
    """Stream ``url`` to ``destination``. No-op when the file already exists.

    Downloads land on a temp path and are renamed on completion, so an
    interrupted transfer never leaves a truncated file that looks cached.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return destination

    tmp = destination.with_suffix(destination.suffix + ".part")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
        progress.add_task(f"Downloading {destination.name}", total=None)
        with httpx.stream("GET", url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as response:
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for block in response.iter_bytes(chunk_size=1 << 16):
                    handle.write(block)
    tmp.replace(destination)
    return destination


def section(heading: str, body: str | None) -> str:
    """Render one ``## heading`` block, or nothing when the body is empty.

    Documents are assembled from these so the structural chunker has real
    boundaries to split on instead of guessing at character offsets.
    """
    if not body or not body.strip():
        return ""
    return f"## {heading}\n{body.strip()}\n\n"
