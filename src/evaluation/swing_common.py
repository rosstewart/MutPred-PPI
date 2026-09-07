#!/usr/bin/env python
"""Vendored SWING blind-test internals (feature encoding + SF/VCFP benchmark loading).

Vendored from (external, non-repo path):
    /home/rcstewart/ppi_lossgain/SWING_scripts/blind_test/run_swing_blind_test_vcfp.py

Kept here so the repo is self-contained (no dependency on an external,
non-versioned path). Only the pieces actually needed by
supplement_swing_vc1pcava.py are vendored: the window-encoding / k-mer /
Doc2Vec-corpus feature pipeline, the SWING-format DataFrame builder, and the
Sahni+Fragoza training-benchmark loader. `_TRAINING_DATA_CSV` is repointed to
the repo's internal training data cache (matching the sed fix applied to the
other supplement_*_vc1pcava.py scripts this session); all other logic is
unchanged from the original.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path

import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import REPO_ROOT  # noqa: E402


_PUB = REPO_ROOT
_TRAINING_DATA_CSV = _PUB / "data_caches" / "training_data_internal.csv"

# ── Grantham aa_score_dict ────────────────────────────────────────────────────

_AA_SCORES = {
    "A": 8.1, "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5, "E": 12.3,
    "Q": 10.5, "G": 9.0, "H": 10.4, "I": 5.2, "L": 4.9, "K": 11.3,
    "M": 5.7, "F": 5.2, "P": 8.0, "S": 9.2, "T": 8.6, "W": 5.4,
    "Y": 6.2, "V": 5.9,
}
_AAs = list(_AA_SCORES.keys())
_AA_SCORE_DICT: dict[str, int] = {}
for _i in range(len(_AAs)):
    for _j in range(len(_AAs) - _i):
        _pair = _AAs[_i] + _AAs[_j + _i]
        _score = round(abs(_AA_SCORES[_AAs[_i]] - _AA_SCORES[_AAs[_j + _i]]))
        _AA_SCORE_DICT[_pair] = _score
        _AA_SCORE_DICT[_pair[::-1]] = _score

# ── Doc2Vec / XGBoost hyper-parameters ───────────────────────────────────────

_D2V_DIM    = 128
_D2V_DM     = 1
_D2V_ALPHA  = 0.08711
_D2V_WINDOW = 6
_D2V_EPOCHS = 52

_XGB_N_EST  = 375
_XGB_DEPTH  = 6
_XGB_LR     = 0.08966


# ── feature helpers ───────────────────────────────────────────────────────────

def _get_window_encodings(df: pd.DataFrame, window_k: int = 1, padding_score: int = 9) -> list[str]:
    encodings = []
    for i in tqdm(df.index, desc="window encodings", file=_sys.stdout):
        pos = df.at[i, "Position"] - 1
        mut_window = df.at[i, "Mutated_Seq"][pos - window_k : pos + window_k + 1]
        interactor = df.at[i, "Interactor_Seq"]
        ppi_enc = ""
        for its in range(len(interactor)):
            window_scores = ""
            for k in range(len(mut_window)):
                try:
                    score = _AA_SCORE_DICT[mut_window[k] + interactor[k + its]]
                except (KeyError, IndexError):
                    score = padding_score
                window_scores += str(score)
            ppi_enc += window_scores
        encodings.append(ppi_enc)
    return encodings


def _get_kmers(encodings: list[str], k: int = 7, padding_score: int = 9) -> list[list[str]]:
    padding = {str(padding_score) * i for i in range(1, k + 1)} | {str(padding_score)}
    result = []
    for enc in tqdm(encodings, desc="k-mers", file=_sys.stdout):
        kmers = [enc[j : j + k] for j in range(len(enc) - k + 1) if enc[j : j + k] not in padding]
        result.append(kmers)
    return result


def _get_corpus(matrix: list[list[str]]):
    for i, doc in enumerate(matrix):
        yield gensim.models.doc2vec.TaggedDocument(doc, [i])


# ── data loading ──────────────────────────────────────────────────────────────

def _build_swing_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        mut = row["mutation"]
        before_aa, pos_1based, after_aa = mut[0], int(mut[1:-1]), mut[-1]
        if before_aa == after_aa:
            continue
        wt_seq  = str(row["interactor_sequence"])
        par_seq = str(row["partner_sequence"])
        if not wt_seq or not par_seq or wt_seq == "nan" or par_seq == "nan":
            continue
        if pos_1based < 1 or pos_1based > len(wt_seq):
            continue
        if wt_seq[pos_1based - 1] != before_aa:
            continue
        vt_seq = wt_seq[: pos_1based - 1] + after_aa + wt_seq[pos_1based:]
        rows.append({
            "interactor":     row["interactor"],
            "partner":        row["partner"],
            "mutation":       mut,
            "Position":       pos_1based,
            "Before_AA":      before_aa,
            "After_AA":       after_aa,
            "Target_Seq":     wt_seq,
            "Interactor_Seq": par_seq,
            "Mutated_Seq":    vt_seq,
            "Type":           "Mutant",
            "Y2H_score":      int(row["perturbed"]),
        })
    return pd.DataFrame(rows).reset_index(drop=True)


def load_benchmark() -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    df = pd.read_csv(_TRAINING_DATA_CSV)
    sf_mask = df["dataset"].str.contains("Sahni") | df["dataset"].str.contains("Fragoza")
    vc_mask = df["dataset"].str.contains("VarChAMP") | df["dataset"].str.contains("VarChAMP_pooled")
    sf_proteins = set(df.loc[sf_mask, "interactor"]) | set(df.loc[sf_mask, "partner"])

    train_raw = df[sf_mask].copy().reset_index(drop=True)
    # Exclude test rows whose (interactor, mutation) appears in SF training (variant-level)
    sf_variants = set(zip(train_raw["interactor"], train_raw["mutation"]))
    test_raw = df[vc_mask].copy()
    test_raw = test_raw[~test_raw.apply(
        lambda r: (r["interactor"], r["mutation"]) in sf_variants, axis=1
    )].reset_index(drop=True)

    train_df = _build_swing_df(train_raw)
    test_df  = _build_swing_df(test_raw)
    print(f"  Train (SF): {len(train_df)}  Test (VCFP): {len(test_df)}", flush=True)
    return train_df, test_df, sf_proteins
