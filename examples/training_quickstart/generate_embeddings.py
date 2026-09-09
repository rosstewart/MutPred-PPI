#!/usr/bin/env python
"""Generate ProtT5 embeddings for wt_and_vt.fasta (produced by
src/inference/01_make_contact_graphs_and_fasta.py) into a single .h5 file,
using the real, unmodified ProtT5 loader utilities from
src/inference/utils/prott5_loader.py (the same ones 02_run_mutpred-ppi_inference.py
uses, just batched over the whole small FASTA at once instead of one variant
at a time, since this is a training set rather than a one-off inference call).
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src" / "inference"))

from utils.prott5_loader import get_T5_model, get_embeddings  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"device: {device}")

    model, vocab = get_T5_model(model_dir=None, device=device)
    print("ProtT5 model loaded")

    fasta_path = _HERE / "wt_and_vt.fasta"
    out_path = _HERE / "wt_and_vt_t5_embs.h5"
    get_embeddings(str(fasta_path), str(out_path), model, vocab, per_protein=False, device=device)
    print(f"Wrote embeddings: {out_path}")


if __name__ == "__main__":
    main()
