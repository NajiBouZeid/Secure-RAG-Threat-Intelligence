"""Embedding inversion: what a stolen vector index gives back as text.

The attack in this package assumes the index itself is the breach -- a stolen
snapshot, or the unauthenticated Qdrant this project runs on localhost -- and
asks how much of the corpus an attacker reconstructs from vectors alone.
"""

from __future__ import annotations

from threatrag.security.inversion.backfill import BackfillStats, backfill_vectors
from threatrag.security.inversion.export import (
    ExportManifest,
    export_targets,
    load_reconstructions,
    load_truth,
    load_vectors,
)
from threatrag.security.inversion.reidentify import (
    ReidentifyReport,
    build_reference,
    reidentify,
)
from threatrag.security.inversion.sample import SampleReport, sample_chunks
from threatrag.security.inversion.score import InversionScores, score_reconstructions

__all__ = [
    "BackfillStats",
    "ExportManifest",
    "InversionScores",
    "ReidentifyReport",
    "SampleReport",
    "backfill_vectors",
    "build_reference",
    "export_targets",
    "load_reconstructions",
    "load_truth",
    "load_vectors",
    "reidentify",
    "sample_chunks",
    "score_reconstructions",
]
