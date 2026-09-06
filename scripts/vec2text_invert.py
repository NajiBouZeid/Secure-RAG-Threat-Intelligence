"""Reconstruct text from stolen GTR vectors. Runs off this machine, on a GPU.

Why it is a loose script and not a package module: vec2text is pinned against a
transformers generation from 2024, and installing it into the project venv
would put the working M1-M4 stack at the mercy of that resolution. It also
needs a GPU in practice -- the corrector runs tens of forward passes per vector
with beam search, which is minutes per vector on a CPU. So the inversion step
runs in a disposable environment (a free Kaggle T4 is enough) and this repo
keeps only the two ends: `threatrag invert prepare` produces the input,
`threatrag invert score` consumes the output. Neither imports vec2text.

Upload only what `prepare` names -- vectors.npy and vector_ids.json. The answer
key stays on the machine that made it; sending it here would make the result
circular.

On Kaggle, in a GPU notebook:

    !pip install -q vec2text
    !python vec2text_invert.py --in /kaggle/input/<dataset> --out /kaggle/working

vec2text declares no upper bound on transformers, so pip will install a current
one and the break, if it comes, is at runtime rather than at resolution time.
If generation raises, pin transformers down (4.44 is a version the corrector was
exercised against) and restart the runtime.

A limitation to read before the numbers. The only public GTR corrector is
`jxm/gtr__nq__32__correct`: trained on 32-token Natural Questions passages. The
chunks in this index are 512 characters, three to four times that. The attack
is being run outside the distribution its corrector was fitted to, which is the
honest position -- it is the corrector a real attacker would have -- but it
means a weak reconstruction is evidence about the available tooling and not
about GTR being hard to invert in principle. Measuring the attack on the length
it was built for is a matter of exporting a truncated bundle from the machine
that holds the text, not of changing anything here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

VECTORS_FILE = "vectors.npy"
IDS_FILE = "vector_ids.json"
OUTPUT_FILE = "reconstructions.jsonl"


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
