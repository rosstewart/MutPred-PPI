"""Shared VarChAMP blind-test array loading.

The blind-test side has no GCV fold structure and no single canonical row
count to check against -- it is a fixed, one-shot held-out evaluation, not a
repeated cross-validation. What it DOES have, and what was unguarded before
this module, is three sibling `.npy` files per (method, class) --
`{method}_c{n}_{preds,labels,vt_ids}.npy` -- that must agree in length with
each other. `blind_test_figures.py` and `restratify_skempi_methods.py` each
loaded these independently with no check at all: if one file were regenerated
and a sibling were not, `preds[i]` and `labels[i]` would silently stop
describing the same variant.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.gcv_common import StaleCacheError

__all__ = ["load_class_arrays"]


def load_class_arrays(method: str, test_class: int, results_dir, *,
                      require_vt_ids: bool = True):
    """`(preds, labels, vt_ids)` for one (method, class), or `None` if absent.

    Raises `StaleCacheError` if the three sibling files exist but disagree in
    length -- the one thing every blind-test consumer must check and none did.
    `vt_ids` is optional (`require_vt_ids=False`) for callers that only need
    preds/labels; when required and the file is absent, this returns `None`
    for the whole triple rather than a two-of-three partial result, since a
    caller that asked for vt_ids needs it to attribute a prediction to a row.
    """
    results_dir = Path(results_dir)
    preds_f = results_dir / f"{method}_c{test_class}_preds.npy"
    labels_f = results_dir / f"{method}_c{test_class}_labels.npy"
    vt_ids_f = results_dir / f"{method}_c{test_class}_vt_ids.npy"

    if not (preds_f.exists() and labels_f.exists()):
        return None
    if require_vt_ids and not vt_ids_f.exists():
        return None

    preds = np.load(preds_f)
    labels = np.load(labels_f)
    vt_ids = np.load(vt_ids_f, allow_pickle=True) if vt_ids_f.exists() else None

    lengths = {"preds": len(preds), "labels": len(labels)}
    if vt_ids is not None:
        lengths["vt_ids"] = len(vt_ids)
    if len(set(lengths.values())) > 1:
        raise StaleCacheError(
            f"{method} c{test_class}: sibling arrays disagree in length "
            f"({lengths}). One of {preds_f.name}/{labels_f.name}/"
            f"{vt_ids_f.name if vt_ids is not None else '(no vt_ids)'} was "
            f"regenerated without the others; predictions and labels can no "
            f"longer be assumed to describe the same row.")

    return preds, labels, vt_ids
