#!/usr/bin/env python3
"""Apply the train/test overlap filter to already-scored blind-test arrays.

Re-running a method to drop leaked test rows is wasted compute for every
predictor whose score for a row depends only on that row: the model is fitted on
`train_df`, which the filter never touches, and inference is per-row, so a
surviving prediction is bit-identical whether or not the dropped rows were
present. Filtering the saved arrays is therefore EXACTLY equivalent, not an
approximation.

This holds for every method here, SWING included. SWING's blind-test mode fits
Doc2Vec on the training rows alone and INFERS each test vector independently
(`_fold_features`), so dropping test rows cannot change a surviving one. Its
"test pretrain" arm is different -- there the test documents are in the Doc2Vec
corpus, so every vector depends on the corpus as a whole -- and that arm must be
re-run rather than filtered.

Each array set is filtered against the training set its own name declares --
`(sahni, ...)` against sahni_only, everything else against sahni_fragoza --
because a row leaked relative to one is not necessarily leaked relative to the
other.

    conda run -n ppi python src/evaluation/apply_blind_test_overlap_filter.py [--mode variant] [--dry-run]
"""
import argparse
import glob
import os
import sys

import numpy as np

from paths import VARCHAMP_BLIND_TEST_DIR
from utils.gcv_common import dataset_config, dataset_name, load_data

# Arms whose vectors depend on the test set as a whole, so filtering is not
# equivalent to re-running. "test pretrain" fits Doc2Vec over train+test.
REFUSE = ("test pretrain",)


def _train_sets():
    out = {}
    for key in ("sahni_only", "sahni_fragoza"):
        tr = load_data(dataset_config(dataset_name(key)))
        out[key] = (set(tr.interactor + " " + tr.partner + " " + tr.mutation),
                    set(zip(tr.interactor, tr.mutation)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["variant", "triplet"], default="variant")
    ap.add_argument("--dir", default=str(VARCHAMP_BLIND_TEST_DIR))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    D = args.dir
    trains = _train_sets()
    methods = sorted({os.path.basename(f).rsplit("_c", 1)[0]
                      for f in glob.glob(f"{D}/*_c[123]_preds.npy")})
    if not methods:
        print(f"no blind-test arrays in {D}", file=sys.stderr)
        return 1

    for m in methods:
        if any(r in m for r in REFUSE):
            print(f"{m}\n    SKIPPED -- corpus-level fit, must be re-run")
            continue
        key = "sahni_only" if "(sahni," in m or "(sahni train)" in m else "sahni_fragoza"
        trip, pair = trains[key]
        kept = dropped = 0
        for c in (1, 2, 3):
            vf = f"{D}/{m}_c{c}_vt_ids.npy"
            if not os.path.exists(vf):
                continue
            v = np.load(vf, allow_pickle=True)
            if len(v) == 0:
                continue
            ids = [str(x) for x in v]
            drop = np.array([i in trip for i in ids])
            if args.mode == "variant":
                drop |= np.array([tuple(i.split(" ")[j] for j in (0, 2)) in pair
                                  for i in ids])
            keep = ~drop
            kept += int(keep.sum()); dropped += int(drop.sum())
            if args.dry_run:
                continue
            for kind in ("preds", "labels", "vt_ids"):
                f = f"{D}/{m}_c{kind and c}_{kind}.npy" if False else f"{D}/{m}_c{c}_{kind}.npy"
                a = np.load(f, allow_pickle=True)
                np.save(f, a[keep])
        print(f"{m}\n    train={key}  kept={kept}  dropped={dropped}"
              f"{'  (dry run)' if args.dry_run else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
