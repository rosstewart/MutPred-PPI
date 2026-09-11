"""Small runtime helpers shared by every script: device and scaler resolution.

Both existed in many hand-rolled copies. Neither is interesting on its own; the
point is that there is now one behaviour instead of several.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["resolve_device", "load_mutation_diff_scaler"]


def resolve_device(requested: str | None = None):
    """A `torch.device`, falling back to CPU when CUDA is unavailable.

    Six spellings of this existed. Three of them -- `mutpred_ppi_gcv.py`,
    `precompute_esm2_datasets.py`, `precompute_prott5_datasets.py` -- were a
    bare `torch.device(args.device)` over a `"cuda:0"` default, which does not
    fail at construction but at the first `.to(device)`, with a CUDA error
    rather than a message a user can act on. For a reproduction package that has
    to run on whatever hardware a reader has, silently degrading to CPU is the
    right behaviour, and saying so is better than either crashing or hiding it.

    An explicit `"cpu"` is always honoured. An explicit CUDA request that cannot
    be satisfied prints a warning and degrades rather than raising, so a long
    batch job started on a machine whose GPU is busy still completes.
    """
    import torch

    want = str(requested) if requested else ("cuda" if torch.cuda.is_available() else "cpu")
    if want.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] {want} requested but CUDA is unavailable -- using CPU. "
              f"This is much slower; pass --device cpu to silence this.",
              flush=True)
        want = "cpu"
    return torch.device(want)


def load_mutation_diff_scaler(models_dir=None):
    """The fitted `mutation_diff_scaler.pkl`, or None if absent.

    Five call sites each built this path themselves (`weights/` via four
    different constants, plus one f-string). Returning None rather than raising
    matches what the callers already did: the scaler is part of the MegaScale
    pretraining pipeline, and a model trained without it must not have one
    applied.
    """
    import joblib
    from paths import WEIGHTS_DIR

    path = Path(models_dir or WEIGHTS_DIR) / "mutation_diff_scaler.pkl"
    return joblib.load(str(path)) if path.exists() else None
