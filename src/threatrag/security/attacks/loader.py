"""Loading attack files and turning them into indexable documents.

The attack corpus lives in ``attacks/`` as one YAML file per case. Loading them
is deliberately the same shape as loading any other source: parse, validate,
hand back objects. The difference is only that the resulting document is forced
to ``SYNTHETIC_ADVERSARIAL`` -- the attacker's content is labelled as attacker
content the moment it enters the index.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from threatrag.domain.models import Document, SourceType
from threatrag.security.attacks.schema import Attack

DEFAULT_ATTACK_DIR = Path("attacks")

# Placeholder in an attack payload replaced with the live sink's base URL at
# ingest time, so an exfiltration beacon points at the port the sink actually
# bound to rather than a hard-coded one that would differ between runs.
SINK_PLACEHOLDER = "{SINK}"


def load_attacks(directory: Path | str = DEFAULT_ATTACK_DIR) -> list[Attack]:
    """Load and validate every ``*.yaml`` attack file, sorted by id.

    Ids must be unique across the corpus: a duplicate would silently overwrite
    one attack's indexed document with another's, since the document id is
    derived from the attack id.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"{directory} missing; expected the attack corpus there.")

    attacks: list[Attack] = []
    for path in sorted(directory.glob("*.yaml")):
        with path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        try:
            attacks.append(Attack.model_validate(payload))
        except Exception as exc:
            raise ValueError(f"{path}: {exc}") from exc

    ids = [attack.id for attack in attacks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate attack id(s): {duplicates}")
    return attacks


def attack_to_document(attack: Attack, *, sink_base: str | None = None) -> Document:
    """Render one attack into the document the pipeline will chunk and index.

    ``source_type`` is not taken from the file: it is always
    ``SYNTHETIC_ADVERSARIAL``. That is the M4 hard constraint made mechanical --
    no attack can enter the index wearing another source's label.
    """
    text = attack.doc.text
    if sink_base is not None:
        text = text.replace(SINK_PLACEHOLDER, sink_base)

    return Document(
        id=f"attack:{attack.id}",
        title=attack.doc.title,
        text=text,
        source_type=SourceType.SYNTHETIC_ADVERSARIAL,
        source_ref=attack.doc.source_ref,
        url=None,
        tlp=attack.doc.tlp,
        trust_tier=attack.doc.trust_tier,
        metadata={
            "attack_id": attack.id,
            "family": attack.family.value,
            "synthetic": "true",
        },
    )
