#!/usr/bin/env python
"""Shared utilities for the VC1p+CAVA VCFP blind-test supplement scripts.

VC1p (VarChAMP 1-protein) and CAVA entries live in the SFVCFP pkl but were
historically excluded from the main VCFP blind test because their graphs use
gene-name/Entrez IDs rather than UniProt IDs. Each per-method supplement
(`supplement_{mutpredppi,esignet,mint,pplm,swing}_vc1pcava.py`) remaps these
entries to UniProt (via gene_symbol_to_uniprot.pkl, the same mapping used in
load_sahni_fragoza_varchamp1p_cava() for GCV construction), classifies them
into C1/C2/C3 against the SF (Sahni+Fragoza) protein set, and writes
`{description}_vc1pcava_c{1,2,3}_{preds,labels,vt_ids}.npy`.

This module factors out the logic that was previously duplicated verbatim
across the 5 supplement scripts:
  - build_sf_proteins() / load_sf_train_df(): SF (Sahni+Fragoza) training rows.
  - classify(): C1/C2/C3 classification given two UniProt IDs.
  - load_vc1pcava_sources() + iter_vc1pcava_entries(): load VC1p/CAVA raw data,
    remap gene-symbol IDs to UniProt, and de-duplicate (22 entries are shared
    between VC1p and CAVA).
  - save_vc1pcava_supplement(): per-class npy save + AUC reporting.

This is a refactor for reuse only — no behavior change relative to the
original per-script copies.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_CV_MOD = Path(__file__).resolve().parent  # src/evaluation
if str(_CV_MOD) not in sys.path:
    sys.path.insert(0, str(_CV_MOD))

from mutpred_ppi_cv import (  # noqa: E402

    _load_varchamp1p_raw, _load_cava_raw,
    get_gene_name, split_wt_id_underscore,
)

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import HOME_DIR, REPO_ROOT  # noqa: E402

_PUB = REPO_ROOT
TRAINING_CSV = _PUB / "data_caches" / "training_data_internal.csv"
VC1P_MAP = HOME_DIR / "varchamp1p" / "gene_symbol_to_uniprot.pkl"
CAVA_MAP = HOME_DIR / "cava" / "gene_symbol_to_uniprot.pkl"
OUT_DIR = _PUB / "results" / "varchamp_seqcnf_newvar_eval"


def build_sf_proteins(training_csv: Path = TRAINING_CSV) -> set[str]:
    """Union of interactor/partner UniProt IDs from Sahni+Fragoza training rows."""
    df = pd.read_csv(training_csv)
    sf_mask = df["dataset"].str.contains("Sahni") | df["dataset"].str.contains("Fragoza")
    return set(df.loc[sf_mask, "interactor"]) | set(df.loc[sf_mask, "partner"])


def load_sf_train_df(training_csv: Path = TRAINING_CSV) -> pd.DataFrame:
    """Sahni+Fragoza training rows only, as a fresh DataFrame."""
    df = pd.read_csv(training_csv)
    sf_mask = df["dataset"].str.contains("Sahni") | df["dataset"].str.contains("Fragoza")
    return df[sf_mask].copy().reset_index(drop=True)


def classify(u1: str, u2: str, sf_proteins: set[str]) -> int:
    """C1: both proteins in SF training set. C2: one. C3: neither."""
    a_in = u1 in sf_proteins
    b_in = u2 in sf_proteins
    if a_in and b_in:
        return 1
    if a_in or b_in:
        return 2
    return 3


def load_uniprot_maps() -> tuple[dict, dict]:
    """Return (vc1p_gs2u, cava_gs2u) gene-symbol -> UniProt maps."""
    vc1p_gs2u = pickle.load(open(VC1P_MAP, "rb"))
    cava_gs2u = pickle.load(open(CAVA_MAP, "rb"))
    return vc1p_gs2u, cava_gs2u


def load_vc1pcava_sources() -> list[tuple[dict, dict]]:
    """Load VC1p + CAVA raw data dicts paired with their UniProt maps.

    Returns [(vc1p_data, vc1p_gs2u), (cava_data, cava_gs2u)], ready to pass
    directly to iter_vc1pcava_entries().
    """
    print("Loading UniProt maps…")
    vc1p_gs2u, cava_gs2u = load_uniprot_maps()

    print("Loading VC1p data…")
    vc1p = _load_varchamp1p_raw()
    print(f"  {len(vc1p['all_vt_ids'])} entries")

    print("Loading CAVA data…")
    cava = _load_cava_raw()
    print(f"  {len(cava['all_vt_ids'])} entries")

    return [(vc1p, vc1p_gs2u), (cava, cava_gs2u)]


def iter_vc1pcava_entries(
    sources: list[tuple[dict, dict]],
    sf_proteins: set[str],
) -> Iterator[tuple[dict, int, str, str, str, str, str, int, int]]:
    """Yield de-duplicated, UniProt-remapped vc1pcava entries.

    sources: list of (data_dict, gene_symbol_to_uniprot_dict) pairs, e.g. from
        load_vc1pcava_sources().
    sf_proteins: SF UniProt protein set, from build_sf_proteins().

    For each (data, gs2u) source, iterates over data["all_vt_ids"], remaps the
    gene-symbol/Entrez wt_id to a UniProt pair, skips entries that can't be
    mapped, and de-duplicates on the remapped UniProt vt_id (22 entries are
    shared between VC1p and CAVA).

    Yields, for each unique entry:
        (data, i, wt_id, variant_0based, u1, u2, uniprot_vt_id, label, class_label)
    where `data`/`i` let the caller pull whatever per-entry fields it needs
    (e.g. data["prott5_embeddings"][i], data["wt_seqs"][i], ...).
    """
    seen: set[str] = set()
    n_ok = n_skip_dup = n_skip_missing_map = 0

    for data, gs2u in sources:
        for i, vt_id in enumerate(data["all_vt_ids"]):
            wt_id = data["all_wt_ids"][i]
            variant = vt_id.split(" ", 1)[1]  # e.g. "V131I" (0-based)

            try:
                a, b = split_wt_id_underscore(wt_id)
                u1 = gs2u[get_gene_name(a)]
                u2 = gs2u[get_gene_name(b)]
            except (KeyError, ValueError):
                n_skip_missing_map += 1
                continue

            uniprot_vt_id = f"{u1} {u2} {variant}"
            if uniprot_vt_id in seen:
                n_skip_dup += 1
                continue
            seen.add(uniprot_vt_id)

            label = 1 if len(data["pos_labels"][i]) > 0 else 0
            c = classify(u1, u2, sf_proteins)
            n_ok += 1

            yield data, i, wt_id, variant, u1, u2, uniprot_vt_id, label, c

    print(f"  ok={n_ok}  dup_skip={n_skip_dup}  map_miss={n_skip_missing_map}")


def save_vc1pcava_supplement(
    scores: Sequence[float],
    labels: Sequence[int],
    vt_ids: Sequence[str],
    classes: Sequence[int],
    description: str,
    out_dir: Path = OUT_DIR,
    filter_nan: bool = True,
) -> str:
    """Save per-class (C1/C2/C3) vc1pcava supplement npy arrays, with AUC reporting.

    scores/labels/vt_ids/classes must be the same length (parallel arrays).
    If filter_nan, entries with NaN scores are dropped first (cache misses in
    some predictors surface as NaN rather than being excluded upstream).

    Returns `description` (the method key used for the saved files, and the
    same key expected by merge_vc1pcava_into_main.merge_method() /
    restratify_vcfp_blind_test.restratify_one_method()).
    """
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels)
    vt_ids = np.asarray(vt_ids)
    classes = np.asarray(classes)

    if filter_nan:
        valid = ~np.isnan(scores)
        n_nan = int((~valid).sum())
        if n_nan:
            print(f"  WARNING: {n_nan}/{len(scores)} NaN scores (cache misses)")
        scores, labels, vt_ids, classes = scores[valid], labels[valid], vt_ids[valid], classes[valid]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{description}_vc1pcava"

    for c in [1, 2, 3]:
        mask = classes == c
        n_pos = int((labels[mask] == 1).sum())
        n_neg = int((labels[mask] == 0).sum())
        if mask.sum() == 0:
            auc_str = "n/a"
        elif n_pos < 2 or n_neg < 2:
            auc_str = "too few"
        else:
            auc_str = f"{roc_auc_score(labels[mask], scores[mask]):.4f}"
        print(f"  C{c}: n={int(mask.sum())} (pos={n_pos}, neg={n_neg}) AUC={auc_str}")
        np.save(out_dir / f"{suffix}_c{c}_preds.npy",  scores[mask].astype(np.float32))
        np.save(out_dir / f"{suffix}_c{c}_labels.npy", labels[mask])
        np.save(out_dir / f"{suffix}_c{c}_vt_ids.npy", vt_ids[mask])

    print(f"Saved → {out_dir}/{suffix}_c{{1,2,3}}_*.npy")
    return description
