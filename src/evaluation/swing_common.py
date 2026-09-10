#!/usr/bin/env python
"""Shared SWING internals: feature encoding, DataFrame builders, Doc2Vec.

Single source for everything SWING-specific in this repo. Used by the
cross-validation driver (`swing_gcv.py`), so the feature pipeline and
hyperparameters cannot drift from a second copy.

Vendored from (external, non-repo path):
    /path/to/upstream-scripts/SWING_scripts/blind_test/run_swing_blind_test_vcfp.py

Vendored from the external SWING scripts so the repo is self-contained. The
window-encoding / k-mer / Doc2Vec-corpus pipeline here was checked against the
external SFVCFP GCV script's own copies on 200 real benchmark rows: the amino-acid
score dictionary, the window encodings, the k-mers and the tagged corpus are all
identical, so the external copies were redundant rather than divergent.
"""
from __future__ import annotations

import sys as _sys
from collections import Counter

import gensim
import pandas as pd
from tqdm import tqdm

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
from paths import REPO_ROOT  # noqa: E402
from utils import mutations  # noqa: E402


_PUB = REPO_ROOT

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

def _build_swing_df(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """SWING-format frame, skipping rows that cannot be mutated -- and counting them.

    The skips are unchanged: SWING has always dropped these rows, and turning any
    of them into a raise would move a published SWING number. What has changed is
    that each drop is now attributed and printed instead of being an anonymous
    `continue`, and the parse/apply go through `utils.mutations` rather than the
    inline `mut[0], int(mut[1:-1]), mut[-1]` that could not reject junk.
    """
    rows = []
    dropped: Counter[str] = Counter()
    for _, row in df.iterrows():
        mut = str(row["mutation"])
        try:
            before_aa, pos_1based, after_aa = mutations.parse(mut)
        except ValueError:
            dropped["unparseable_mutation"] += 1
            continue
        if before_aa == after_aa:
            dropped["synonymous"] += 1
            continue
        wt_seq  = str(row["interactor_sequence"])
        par_seq = str(row["partner_sequence"])
        if not wt_seq or not par_seq or wt_seq == "nan" or par_seq == "nan":
            dropped["missing_sequence"] += 1
            continue
        vt_seq = mutations.apply(wt_seq, mut)
        if vt_seq is None:
            # `apply` folds the old out-of-range and WT-mismatch checks into one
            # None; separate them here so the report keeps naming both.
            dropped["position_out_of_range" if pos_1based > len(wt_seq)
                    else "wt_residue_mismatch"] += 1
            continue
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
    if dropped:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(dropped.items()))
        print(f"  _build_swing_df{f' [{label}]' if label else ''}: "
              f"kept {len(rows)}/{len(df)}, dropped {sum(dropped.values())} "
              f"({detail})", flush=True)
    return pd.DataFrame(rows).reset_index(drop=True)


def build_wt_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a SWING-format df with `Mutated_Seq` reverted to wild type.

    Doc2Vec is trained on mutant and wild-type sequences together, so each row
    needs its WT counterpart. Raises rather than silently correcting when the
    sequence does not carry the expected mutant residue -- a mismatch means the
    row's sequence and its `Position`/`After_AA` annotation disagree, which would
    otherwise produce a quietly wrong WT sequence.
    """
    wt_seqs = []
    for i in df.index:
        seq   = df.at[i, "Mutated_Seq"]
        bef   = df.at[i, "Before_AA"]
        aft   = df.at[i, "After_AA"]
        pos_0 = int(df.at[i, "Position"]) - 1
        if seq[pos_0] != aft:
            raise ValueError(
                f"After_AA mismatch at index {i}: sequence has {seq[pos_0]!r} "
                f"at position {pos_0 + 1}, annotation says {aft!r}"
            )
        wt_seqs.append(seq[:pos_0] + bef + seq[pos_0 + 1:])
    out = df.copy()
    out["Mutated_Seq"] = wt_seqs
    out["Type"] = "WildType"
    return out


def build_d2v(df_combined: pd.DataFrame):
    """Train Doc2Vec on a combined mutant+wild-type frame.

    Hyperparameters are upstream SWING's and must not be tuned -- see
    docs/METHOD_PROVENANCE.md.
    """
    from gensim.models.doc2vec import Doc2Vec

    encodings = _get_window_encodings(df_combined)
    kmers     = _get_kmers(encodings)
    corpus    = list(_get_corpus(kmers))
    model = Doc2Vec(vector_size=_D2V_DIM, min_count=1, alpha=_D2V_ALPHA,
                    dm=_D2V_DM, window=_D2V_WINDOW)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=_D2V_EPOCHS)
    return model
