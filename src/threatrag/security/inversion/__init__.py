"""Embedding inversion: what a stolen vector index gives back as text.

The attack in this package assumes the index itself is the breach -- a stolen
snapshot, or the unauthenticated Qdrant this project runs on localhost -- and
asks how much of the corpus an attacker reconstructs from vectors alone.
"""

from __future__ import annotations

from threatrag.security.inversion.sample import SampleReport, sample_chunks

__all__ = ["SampleReport", "sample_chunks"]
