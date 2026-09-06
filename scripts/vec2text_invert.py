"""Reconstruct text from stolen GTR vectors. Needs a GPU and its own venv.

Why it is a loose script and not a package module: vec2text is built against a
2024 transformers and installing it beside the project would put the working
M1-M5 stack at the mercy of that resolution. It also needs a GPU in practice --
the corrector runs tens of forward passes per vector with beam search. So the
repo keeps only the two ends, and neither imports vec2text: `threatrag invert
prepare` produces the input, `threatrag invert score` consumes the output.

The M5 run that produced reports/m5_inversion.md used a local RTX 4060 through
an isolated venv. Four things had to be worked around to get there, none of
them about inversion, all of them recorded because none is obvious:

    python -m venv .venv-inv
    .venv-inv/Scripts/python -m pip install torch --index-url \\
        https://download.pytorch.org/whl/cu128          # 2.86 GB, ~9 GB installed

    # rouge_score ships sdist-only and builds an invalid wheel name under
    # modern setuptools, which fails the whole vec2text install.
    .venv-inv/Scripts/python -m pip install setuptools wheel
    .venv-inv/Scripts/python -m pip install rouge-score --no-build-isolation
    .venv-inv/Scripts/python -m pip install vec2text

    # transformers 5 initialises models on a meta device, and vec2text's
    # InversionModel.__init__ calls from_pretrained *inside* that context,
    # which transformers 5 forbids outright. Pin the era it was written for.
    .venv-inv/Scripts/python -m pip install "transformers==4.44.2" \\
        "huggingface_hub<1.0" "tokenizers>=0.19,<0.20" \\
        "sentence-transformers==3.0.1" "datasets<3" "accelerate<1.0"

    .venv-inv/Scripts/python scripts/vec2text_invert.py \\
        --in data/inversion --out data/inversion --batch 4

The fourth is Windows-only and handled in code below: vec2text imports the
Unix-only `resource` module at package import.

Use `--batch 4`. Measured on an 8 GB card, batches of 16 and 32 run out of
memory and 8 is *slower* than 4 -- the bottleneck is the sequential correction
steps, so batching buys nothing and costs VRAM. Budget ~43 min per 334-vector
bundle. Do not speed it up by lowering `--steps` or `--beam`: that changes what
is measured and breaks comparability between bundles.

A free cloud GPU works equally well; the script only needs `--in` to hold
vectors.npy and vector_ids.json. If you run it somewhere else, send **only**
those two files. The answer key stays on the machine that made it, because
handing the reconstruction step the source text would make the result circular.

A limitation to read before the numbers. The only public GTR corrector is
`jxm/gtr__nq__32__correct`: trained on 32-token Natural Questions passages. The
chunks in this index are 512 characters, three to four times that. The attack
is being run outside the distribution its corrector was fitted to, which is the
honest position -- it is the corrector a real attacker would have -- but it
means a weak reconstruction is evidence about the available tooling and not
about GTR being hard to invert in principle. `threatrag invert prepare` writes
a 32-token control bundle alongside the main one for exactly this reason.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

VECTORS_FILE = "vectors.npy"
IDS_FILE = "vector_ids.json"
OUTPUT_FILE = "reconstructions.jsonl"


def stub_resource_module() -> None:
    """Make vec2text importable on Windows.

    vec2text's package import reaches `experiments.py`, which imports the
    Unix-only stdlib module `resource`, so `import vec2text` fails outright on
    Windows before any model is touched. It is the only Unix-only import in the
    package and it is used in exactly one place -- raising the core-dump limit
    inside a training entrypoint this script never calls.

    So the stub carries the three names that line touches and does nothing.
    This is narrow on purpose: it removes an import-time platform assumption,
    not a behaviour. If a future vec2text uses `resource` for something real,
    the call lands on a no-op that returns a plausible value, so the guard
    below asserts the module is genuinely absent rather than shadowing a real
    one on a platform that has it.
    """
    if sys.platform != "win32" or "resource" in sys.modules:
        return

    stub = types.ModuleType("resource")
    stub.RLIMIT_CORE = 4  # type: ignore[attr-defined]
    stub.RLIM_INFINITY = -1  # type: ignore[attr-defined]
    stub.setrlimit = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    stub.getrlimit = lambda *args, **kwargs: (-1, -1)  # type: ignore[attr-defined]
    sys.modules["resource"] = stub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_dir", type=Path, required=True)
    parser.add_argument("--out", dest="out_dir", type=Path, required=True)
    parser.add_argument("--corrector", default="gtr-base")
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Correction rounds. The paper's headline numbers use 20 plus beam search.",
    )
    parser.add_argument(
        "--beam",
        type=int,
        default=4,
        help="Sequence beam width. 0 disables beam search, which is much faster and much worse.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Vectors per call. Lower it if the GPU runs out of memory.",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Invert only the first N vectors; 0 means all."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch

    stub_resource_module()
    import vec2text

    ids: list[str] = json.loads((args.in_dir / IDS_FILE).read_text(encoding="utf-8"))
    matrix = np.load(args.in_dir / VECTORS_FILE)
    if args.limit:
        ids, matrix = ids[: args.limit], matrix[: args.limit]
    print(f"{len(ids)} vectors, {matrix.shape[1]}d, corrector={args.corrector}")

    if not torch.cuda.is_available():
        print("WARNING: no GPU visible. This will take hours rather than minutes.")

    corrector = vec2text.load_pretrained_corrector(args.corrector)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / OUTPUT_FILE
    started = time.time()
    done = 0

    # Written incrementally: a session that dies at vector 300 of 334 should
    # still yield 300 usable reconstructions rather than nothing.
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for start in range(0, len(ids), args.batch):
            block = matrix[start : start + args.batch]
            embeddings = torch.tensor(block, dtype=torch.float32)
            if torch.cuda.is_available():
                embeddings = embeddings.cuda()

            texts = vec2text.invert_embeddings(
                embeddings=embeddings,
                corrector=corrector,
                num_steps=args.steps,
                sequence_beam_width=args.beam,
            )

            for offset, text in enumerate(texts):
                handle.write(
                    json.dumps(
                        {"chunk_id": ids[start + offset], "reconstruction": text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            handle.flush()

            done += len(texts)
            rate = done / max(time.time() - started, 1e-6)
            print(f"  {done}/{len(ids)}  ({rate:.2f} vectors/s)", flush=True)

    elapsed = time.time() - started
    print(f"wrote {out_path} -- {done} reconstructions in {elapsed / 60:.1f} min")
    print("Bring this file back and run: threatrag invert score reconstructions.jsonl")


if __name__ == "__main__":
    main()
