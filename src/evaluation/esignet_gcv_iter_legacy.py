#!/usr/bin/env python
"""eSIG-Net group cross-validation training script.

Mirrors the four SWING GCV scripts (SWING_MutInt_Notebook_*_gcv_iter.py) but
replaces Doc2Vec + XGBoost with eSIG-Net (dual-channel SDNN + ESM-2 discriminator).

Key difference from SWING: no STRINGENT_PRETRAIN mode — eSIG-Net always trains
from scratch on each fold's training data only.

Usage:
    conda run -n ppi python esignet_gcv_iter.py --dataset sahni_fragoza --device cuda:0
    conda run -n ppi python esignet_gcv_iter.py --dataset sahni_fragoza_varchamp1p_cava \\
        --device cuda:0 --n-gcv 30 --outdir /path/to/results/
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.metrics import roc_auc_score

# ── eSIG-Net import ────────────────────────────────────────────────────────────
# Resolve benchmark dir relative to this script's location
_BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "2026/mutppi/benchmark"
sys.path.insert(0, str(_BENCHMARK_DIR))

import predictors.esignet as _esignet_mod                                    # noqa: E402
from predictors.esignet import ESigNetPredictor, _compute_573, _FEAT_CACHE  # noqa: E402
from predictors.nn_base import load_cache                                    # noqa: E402

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import CV_DIR, DATA_ROOT  # noqa: E402


# Surface load_cache's INFO logs ("Loading cache: …", "Loaded N entries").
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── fixed paths ───────────────────────────────────────────────────────────────
_CV_DIR = CV_DIR
_FASTA_ROOT = DATA_ROOT
_UNIMAP_DIR  = _FASTA_ROOT / "varchamp1p"


# ── dataset configuration ─────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    name: str
    pos_file: str              # filename relative to --data-root
    neg_file: str
    extra_files: List[str]     # files relative to --data-root; each has Y2H_score column
    fasta_file: str            # filename relative to _FASTA_ROOT
    fasta_use_description: bool  # True → record.description.strip(); False → record.id.replace('|',' ')
    apply_uniprot_map: bool    # remap gene symbols to UniProt IDs before sequence lookup
    fold_splits_pat: str       # pattern with {seed} placeholder
    all_vt_ids_file: str       # file in _CV_DIR (non-seed canonical ordering)
    vt_id_sep: str             # separator between refseq_id and partner in vt_id key
    pair_test_classes_pat: str # pattern with {seed} placeholder
    dedup: bool = False        # sort Y2H_score desc + drop_duplicates on non-label cols
    vt_id_sep_alt: str = ""    # fallback separator tried if vt_id_sep lookup fails (for mixed-sep datasets)


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "sahni": DatasetConfig(
        name="sahni",
        pos_file="interaction_loss.pos",
        neg_file="interaction_loss.neg",
        extra_files=[],
        fasta_file=f"{_FASTA_ROOT}/home/sahni/interaction_loss_wt_and_vt.fasta",
        fasta_use_description=True,
        apply_uniprot_map=False,
        fold_splits_pat="fold_splits_{seed}.pkl",
        all_vt_ids_file="all_vt_ids.pkl",
        vt_id_sep="_",
        pair_test_classes_pat="pair_test_classes_{seed}.npy",
        dedup=False,
    ),
    "sahni_fragoza": DatasetConfig(
        name="sahni_fragoza",
        pos_file="swing_train.pos",
        neg_file="swing_train.neg",
        extra_files=[],
        fasta_file="swing_train/swing_train_wt_and_vt.fasta",
        fasta_use_description=True,
        apply_uniprot_map=False,
        fold_splits_pat="sahni_fragoza_train_fold_splits_{seed}.pkl",
        all_vt_ids_file="sahni_fragoza_train_all_vt_ids.pkl",
        vt_id_sep="-",  # NOTE: hyphen, not underscore
        pair_test_classes_pat="swing_train_pair_test_classes_{seed}.npy",
        dedup=False,
    ),
    "sahni_fragoza_varchamp1p_cava": DatasetConfig(
        name="sahni_fragoza_varchamp1p_cava",
        pos_file="swing_train.pos",
        neg_file="swing_train.neg",
        extra_files=["cava_variant_interaction.tsv", "varchamp1p_variant_interaction.tsv"],
        fasta_file="sahni_fragoza_varchamp1p_cava_wt_and_vt.fasta",
        fasta_use_description=False,
        apply_uniprot_map=True,
        fold_splits_pat="sahni_fragoza_varchamp1p_cava_train_fold_splits_{seed}.pkl",
        all_vt_ids_file="sahni_fragoza_varchamp1p_cava_train_all_vt_ids.pkl",
        vt_id_sep="_",
        pair_test_classes_pat="combined_sahni_fragoza_varchamp1p_cava_seq_confirmed_pair_test_classes_{seed}.npy",
        dedup=True,
    ),
    "sahni_varchamp1p_cava": DatasetConfig(
        name="sahni_varchamp1p_cava",
        pos_file="interaction_loss.pos",
        neg_file="interaction_loss.neg",
        extra_files=["cava_variant_interaction.tsv", "varchamp1p_variant_interaction.tsv"],
        fasta_file="sahni_varchamp1p_cava_wt_and_vt.fasta",
        fasta_use_description=False,
        apply_uniprot_map=False,
        fold_splits_pat="sahni_varchamp1p_cava_train_fold_splits_{seed}.pkl",
        all_vt_ids_file="sahni_varchamp1p_cava_train_all_vt_ids.pkl",
        vt_id_sep="_",
        pair_test_classes_pat="combined_sahni_varchamp1p_cava_seq_confirmed_concat_clust_pair_test_classes_{seed}.npy",
        dedup=True,
    ),
    "sahni_fragoza_varchamp2026": DatasetConfig(
        name="sahni_fragoza_varchamp2026",
        pos_file="swing_train.pos",
        neg_file="swing_train.neg",
        extra_files=["varchamp2026_variant_interaction.tsv"],
        fasta_file="sahni_fragoza_varchamp2026_wt_and_vt.fasta",
        fasta_use_description=False,
        apply_uniprot_map=False,
        fold_splits_pat="sahni_fragoza_varchamp2026_train_fold_splits_{seed}.pkl",
        all_vt_ids_file="sahni_fragoza_varchamp2026_train_all_vt_ids.pkl",
        vt_id_sep="_",
        pair_test_classes_pat="combined_sahni_fragoza_varchamp2026_pair_test_classes_{seed}.npy",
        dedup=True,
    ),
    # SF + VarChAMP_pooled (Jul 2026).  The combined all_vt_ids uses _ for the SF
    # portion (from the split generator) and - for pooled (after normalization in
    # generate_pooled_splits.py), so we try _ first and fall back to -.
    "sahni_fragoza_varchamp_pooled": DatasetConfig(
        name="sahni_fragoza_varchamp_pooled",
        pos_file="swing_train.pos",
        neg_file="swing_train.neg",
        extra_files=["varchamp_pooled_variant_interaction.tsv"],
        fasta_file="sahni_fragoza_varchamp_pooled_wt_and_vt.fasta",
        fasta_use_description=True,
        apply_uniprot_map=False,
        fold_splits_pat="sahni_fragoza_varchamp_pooled_train_fold_splits_{seed}.pkl",
        all_vt_ids_file="sahni_fragoza_varchamp_pooled_train_all_vt_ids.pkl",
        vt_id_sep="_",
        vt_id_sep_alt="-",
        pair_test_classes_pat="combined_sahni_fragoza_varchamp_pooled_pair_test_classes_{seed}.npy",
        dedup=True,
    ),
    # SF + VC2026 + VC1p + CAVA + Pooled (full combined).  Same mixed-sep issue.
    "sahni_fragoza_varchamp_full_pooled": DatasetConfig(
        name="sahni_fragoza_varchamp_full_pooled",
        pos_file="swing_train.pos",
        neg_file="swing_train.neg",
        extra_files=[
            "varchamp2026_variant_interaction.tsv",
            "cava_variant_interaction.tsv",
            "varchamp1p_variant_interaction.tsv",
            "varchamp_pooled_variant_interaction.tsv",
        ],
        fasta_file="sahni_fragoza_varchamp_full_pooled_wt_and_vt.fasta",
        fasta_use_description=True,
        apply_uniprot_map=False,
        fold_splits_pat="sahni_fragoza_varchamp_full_pooled_train_fold_splits_{seed}.pkl",
        all_vt_ids_file="sahni_fragoza_varchamp_full_pooled_train_all_vt_ids.pkl",
        vt_id_sep="_",
        vt_id_sep_alt="-",
        pair_test_classes_pat="combined_sahni_fragoza_varchamp_full_pooled_pair_test_classes_{seed}.npy",
        dedup=True,
    ),
}


# ── data loading ──────────────────────────────────────────────────────────────

def _get_gene_name(gene_name_and_orf_id: str) -> str:
    """Strip numeric ORF suffix from gene+ORF IDs (mirrors SWING helper)."""
    assert not gene_name_and_orf_id.startswith("NP_"), \
        f"_get_gene_name called with RefSeq ID: {gene_name_and_orf_id}"
    if "_" not in gene_name_and_orf_id:
        return gene_name_and_orf_id
    return "_".join(gene_name_and_orf_id.split("_")[:-1])


def _load_uniprot_map() -> dict:
    with open(_UNIMAP_DIR / "varchamp1p_gene_symbol_to_uniprot.pkl", "rb") as f:
        m1 = pickle.load(f)
    with open(_UNIMAP_DIR / "cava_gene_symbol_to_uniprot.pkl", "rb") as f:
        m2 = pickle.load(f)
    return {**m1, **m2}


def _parse_fasta(fasta_path: Path, use_description: bool) -> dict:
    """Parse a FASTA file into a dict keyed by refseq_id or refseq_id+space+variant.

    Variant keys are stored 0-based in the FASTA; we convert to 1-based here so
    they match the Mutation column in the DataFrame.
    """
    raw: dict[str, str] = {}
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        if use_description:
            key = record.description.strip()
        else:
            key = record.id.replace("|", " ")
        raw[key] = str(record.seq)

    # Convert 0-based position in variant keys to 1-based
    converted: dict[str, str] = {}
    for key, seq in raw.items():
        if " " in key:
            protein_id, variant_0based = key.split(" ", 1)
            wt_aa  = variant_0based[0]
            pos_0  = int(variant_0based[1:-1])
            mut_aa = variant_0based[-1]
            new_key = f"{protein_id} {wt_aa}{pos_0 + 1}{mut_aa}"
            converted[new_key] = seq
        else:
            converted[key] = seq
    return converted


def load_data(cfg: DatasetConfig, data_root: Path) -> pd.DataFrame:
    """Load and process the dataset, returning a DataFrame in SWING column format.

    Columns: refseq_id, partner, Mutation, Y2H_score,
             Target_Seq, Interactor_Seq, Before_AA, After_AA, Position
    """
    # 1. Load pos/neg
    pos_df = pd.read_csv(data_root / cfg.pos_file, header=None, sep="\t")
    pos_df.columns = ["refseq_id", "Mutation", "partner"]
    neg_df = pd.read_csv(data_root / cfg.neg_file, header=None, sep="\t")
    neg_df.columns = ["refseq_id", "Mutation", "partner"]
    df = pd.concat([pos_df, neg_df], ignore_index=True)
    df["Y2H_score"] = [1] * len(pos_df) + [0] * len(neg_df)

    # 2. Append extra files (cava, varchamp1p)
    for extra_file in cfg.extra_files:
        extra = pd.read_csv(data_root / extra_file, header=None, sep="\t")
        extra.columns = ["refseq_id", "Mutation", "partner", "Y2H_score"]
        df = pd.concat([df, extra], ignore_index=True)

    # 3. Apply gene_symbol → UniProt mapping
    if cfg.apply_uniprot_map:
        unimap = _load_uniprot_map()

        def _update_id(value: str) -> str:
            # RefSeq IDs (sahni rows) are not in the gene→UniProt map and have
            # no orf-id suffix to strip — pass them through unchanged.
            if value.startswith("NP_"):
                return value
            gene = _get_gene_name(value)
            return unimap.get(gene, value)

        df["refseq_id"] = df["refseq_id"].apply(_update_id)
        df["partner"]   = df["partner"].apply(_update_id)

    # 4. Parse FASTA (positions already 1-based after _parse_fasta)
    fasta_dict = _parse_fasta(_FASTA_ROOT / cfg.fasta_file, cfg.fasta_use_description)

    # 5. Lookup sequences, skip rows with missing seqs or synonymous mutations
    indices_to_drop = []
    new_info = []
    for i, row in df.iterrows():
        refseq_id = row["refseq_id"]
        mutation  = row["Mutation"]
        partner   = row["partner"]
        vt_id     = f"{refseq_id} {mutation}"

        if refseq_id not in fasta_dict or vt_id not in fasta_dict or partner not in fasta_dict:
            indices_to_drop.append(i)
            new_info.append((None,) * 7)
            continue

        wt_seq  = fasta_dict[refseq_id]
        mt_seq  = fasta_dict[vt_id]
        partner_seq = fasta_dict[partner]

        before_aa = mutation[0]
        position  = int(mutation[1:-1])
        after_aa  = mutation[-1]

        if before_aa == after_aa:
            indices_to_drop.append(i)
            new_info.append((None,) * 7)
            continue

        assert wt_seq[position - 1] == before_aa and mt_seq[position - 1] == after_aa, (
            f"{vt_id}: expected {before_aa} at 1-based pos {position}, "
            f"got wt={wt_seq[position-1]} mt={mt_seq[position-1]}"
        )
        new_info.append((before_aa, position, after_aa, wt_seq, partner_seq, mt_seq, "Mutant"))

    new_cols = pd.DataFrame(
        new_info,
        columns=["Before_AA", "Position", "After_AA", "Target_Seq",
                 "Interactor_Seq", "Mutated_Seq", "Type"],
    )
    df = pd.concat([df, new_cols], axis=1)

    if indices_to_drop:
        df = df.drop(indices_to_drop).reset_index(drop=True)

    df["Position"] = df["Position"].astype(int)

    # 6. Deduplicate (datasets that merge multiple sources)
    if cfg.dedup:
        df = df.drop_duplicates()
        cols_to_check = [c for c in df.columns if c != "Y2H_score"]
        df = (df.sort_values("Y2H_score", ascending=False)
                .drop_duplicates(subset=cols_to_check, keep="first"))

    return df


# ── vt_ids alignment ──────────────────────────────────────────────────────────

def _vt_ids_to_1based(raw_ids: list) -> list:
    """Convert stored (0-based) vt_ids to 1-based, matching the DataFrame Mutation column."""
    result = []
    for vt_id in raw_ids:
        prefix, variant = vt_id.split(" ", 1)
        pos_1based = int(variant[1:-1]) + 1
        result.append(f"{prefix} {variant[0]}{pos_1based}{variant[-1]}")
    return result


def align_to_vt_ids(
    df: pd.DataFrame,
    cfg: DatasetConfig,
) -> pd.DataFrame:
    """Re-order df to match the canonical vt_ids ordering stored in all_vt_ids.pkl.

    Returns ordered_df aligned to the vt_ids index.
    """
    with open(_CV_DIR / cfg.all_vt_ids_file, "rb") as f:
        raw_vt_ids = pickle.load(f)
    vt_ids = _vt_ids_to_1based(raw_vt_ids)
    vt_ids_set = set(vt_ids)

    index_mapping = []
    valid_indices = []

    for i, row in df.iterrows():
        vt_id = f"{row['refseq_id']}{cfg.vt_id_sep}{row['partner']} {row['Mutation']}"
        if vt_id not in vt_ids_set and cfg.vt_id_sep_alt:
            vt_id = f"{row['refseq_id']}{cfg.vt_id_sep_alt}{row['partner']} {row['Mutation']}"
        if vt_id in vt_ids_set:
            index_mapping.append(vt_ids.index(vt_id))
            valid_indices.append(i)

    if len(valid_indices) != len(vt_ids):
        n_missing = len(vt_ids) - len(valid_indices)
        print(
            f"WARNING: {n_missing}/{len(vt_ids)} vt_ids not matched in dataset "
            f"({len(valid_indices)} matched). Unmatched rows will be NaN.",
            flush=True,
        )
    assert len(index_mapping) == len(set(index_mapping)), "Duplicate vt_id mappings"

    df = df.loc[valid_indices].reset_index(drop=True)
    # Build full-size ordered_df (len = len(vt_ids)); unmatched positions stay NaN
    ordered_df = pd.DataFrame(index=range(len(vt_ids)), columns=df.columns)
    for old_idx, new_idx in enumerate(index_mapping):
        ordered_df.iloc[new_idx] = df.iloc[old_idx]
    return ordered_df.reset_index(drop=True)


# ── format conversion ─────────────────────────────────────────────────────────

def swing_to_esignet(swing_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a SWING-format slice to eSIG-Net's expected column layout.

    SWING column   →  eSIG-Net column
    refseq_id      →  interactor
    partner        →  partner         (unchanged)
    Mutation       →  mutation        (1-based, already correct)
    Y2H_score      →  perturbed = Y2H_score  (1=disrupted → 1=perturbed)
    Target_Seq     →  interactor_sequence
    Interactor_Seq →  partner_sequence
    """
    out = pd.DataFrame({
        "interactor":          swing_df["refseq_id"].values,
        "partner":             swing_df["partner"].values,
        "mutation":            swing_df["Mutation"].values,
        "perturbed":           swing_df["Y2H_score"].astype(int).values,
        "interactor_sequence": swing_df["Target_Seq"].values,
        "partner_sequence":    swing_df["Interactor_Seq"].values,
    })
    return out.reset_index(drop=True)


# ── feature pre-warming ───────────────────────────────────────────────────────

def _zero_based_variant(mutation: str) -> str:
    """'E80K' → 'E79K' (mirrors esignet._zero_based_variant)."""
    pos1 = int(mutation[1:-1])
    return f"{mutation[0]}{pos1 - 1}{mutation[-1]}"


def audit_esm_cache(
    ordered_df: pd.DataFrame,
    cfg: DatasetConfig,
    min_hit_rate: float,
    require: bool,
) -> None:
    """Load the active ESM cache and report WT/MT key hit rates.

    Aborts when require=True and the mutant-key hit rate falls below
    min_hit_rate. Without a usable cache eSIG-Net's discriminator branch
    silently degenerates to zero-vectors — fail fast instead.
    """
    path = _esignet_mod._ESM_CACHE_PATH
    print(f"\n── ESM-2 cache audit ──", flush=True)
    print(f"path: {path}", flush=True)

    cache = load_cache(path)
    if cache is None:
        msg = (
            f"ESM cache not found at {path}. eSIG-Net's discriminator branch "
            f"will receive all-zero embeddings — model is effectively neutered."
        )
        if require:
            raise SystemExit("ABORT: " + msg + "\nPass --require-esm=false to override.")
        print("WARNING: " + msg, flush=True)
        return

    print(f"entries: {len(cache)}", flush=True)

    # Filter out NaN rows (unmatched vt_ids produce NaN mutations)
    valid_df    = ordered_df.dropna(subset=["Mutation"])
    interactors = valid_df["refseq_id"].astype(str)
    mutations   = valid_df["Mutation"].astype(str)
    unique_wt   = set(interactors)
    unique_mt   = set(zip(interactors, mutations))

    wt_hits = sum(1 for k in unique_wt if k in cache)
    mt_hits_1 = sum(1 for ia, mu in unique_mt if f"{ia}_{mu}" in cache)
    mt_hits_0 = sum(1 for ia, mu in unique_mt
                    if f"{ia}_{_zero_based_variant(mu)}" in cache)

    one_based = mt_hits_1 >= mt_hits_0
    mt_hits = mt_hits_1 if one_based else mt_hits_0
    n_wt, n_mt = len(unique_wt), len(unique_mt)

    print(f"interactor (WT) keys: {wt_hits}/{n_wt} hit "
          f"({wt_hits / max(n_wt, 1):.1%})", flush=True)
    print(f"mutant keys ({'1' if one_based else '0'}-based): "
          f"{mt_hits}/{n_mt} hit ({mt_hits / max(n_mt, 1):.1%})  "
          f"[1-based={mt_hits_1}, 0-based={mt_hits_0}]", flush=True)

    mt_rate = mt_hits / max(n_mt, 1)
    if mt_rate < min_hit_rate:
        msg = (
            f"Mutant ESM hit rate {mt_rate:.1%} is below "
            f"--min-esm-hit-rate {min_hit_rate:.1%}. "
            f"Cache likely keyed by the wrong ID type "
            f"(e.g. UniProt cache vs RefSeq dataset). "
            f"Run precompute_esm_embeddings.py --dataset {cfg.name} "
            f"and pass its output via --esm-cache PATH."
        )
        if require:
            raise SystemExit("ABORT: " + msg)
        print("WARNING: " + msg, flush=True)


def prewarm_features(ordered_df: pd.DataFrame) -> None:
    """Pre-populate the module-level _FEAT_CACHE for all unique sequences.

    eSIG-Net computes 573-dim AAC+CTD features on-the-fly and caches them by
    sequence string. Pre-warming here avoids redundant computation across folds.
    """
    unique_seqs = set()
    for col in ("Target_Seq", "Interactor_Seq"):
        unique_seqs.update(ordered_df[col].dropna().unique())
    print(f"Pre-computing 573-dim features for {len(unique_seqs)} unique sequences...",
          flush=True)
    for seq in unique_seqs:
        _compute_573(seq)
    print(f"  done ({len(_FEAT_CACHE)} sequences cached)", flush=True)


# ── GCV evaluation ────────────────────────────────────────────────────────────

def _compute_class_aucs(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    pair_test_classes: np.ndarray,
    fold_n_test: list,
) -> tuple[list, np.ndarray, dict]:
    """Compute micro and weighted-macro AUROCs by pair_test_class (1/2/3).

    Returns:
        micro_auc         list of 3 AUROCs (over all samples per class)
        macro_auc         np.ndarray shape (3,) weighted per-fold mean
        fold_results      dict of per-fold results
    """
    valid_mask = ~np.isnan(all_preds)
    micro_auc = []
    for ptc in (1, 2, 3):
        mask = (pair_test_classes == ptc) & valid_mask
        print(f"  c{ptc}: {mask.sum()} preds (valid)", flush=True)
        if mask.sum() > 0 and len(np.unique(all_labels[mask])) > 1:
            auc = roc_auc_score(all_labels[mask], all_preds[mask])
        else:
            auc = float("nan")
        micro_auc.append(auc)

    class_auc_avgs = np.zeros(3)
    class_counts   = np.zeros(3, dtype=int)
    fold_results: dict = {}

    curr_idx = 0
    for fold, n_test in enumerate(fold_n_test):
        preds  = all_preds[curr_idx:curr_idx + n_test]
        labels = all_labels[curr_idx:curr_idx + n_test]
        ptcs   = pair_test_classes[curr_idx:curr_idx + n_test]
        curr_idx += n_test

        fold_res = {}
        print(f"\nfold {fold}", flush=True)
        fold_valid = ~np.isnan(preds)
        for ptc in (1, 2, 3):
            mask   = (ptcs == ptc) & fold_valid
            cp     = preds[mask]
            cl     = labels[mask]
            n_pos  = int((cl == 1).sum())
            n_neg  = int((cl == 0).sum())
            fold_res[f"class_{ptc}"] = {"preds": cp, "labels": cl, "auc": None}

            if n_pos > 0 and n_neg > 0:
                auc = roc_auc_score(cl, cp)
                fold_res[f"class_{ptc}"]["auc"] = auc
                class_auc_avgs[ptc - 1] += len(cp) * auc
                class_counts[ptc - 1]   += len(cp)
                print(f"  c{ptc} (n={len(cp)}): AUC-ROC={auc:.4f}", flush=True)
            else:
                print(f"  c{ptc} (n={len(cp)}): SKIPPED (pos={n_pos}, neg={n_neg})",
                      flush=True)

        fold_results[fold] = fold_res

    macro_auc = np.where(class_counts > 0, class_auc_avgs / class_counts, np.nan)
    return micro_auc, macro_auc, fold_results


# ── main GCV loop ─────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    if args.esm_cache:
        _esignet_mod._ESM_CACHE_PATH = args.esm_cache
        print(f"ESM cache path overridden: {args.esm_cache}", flush=True)

    cfg = DATASET_CONFIGS[args.dataset]
    data_root = Path(args.data_root)
    outdir    = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── load and align dataset ────────────────────────────────────────────────
    print(f"Loading dataset: {cfg.name}", flush=True)
    df = load_data(cfg, data_root)
    print(f"  {len(df)} rows after filtering", flush=True)

    ordered_df = align_to_vt_ids(df, cfg)
    print(f"  {len(ordered_df)} rows after vt_ids alignment", flush=True)

    audit_esm_cache(ordered_df, cfg, args.min_esm_hit_rate, args.require_esm)

    prewarm_features(ordered_df)

    labels = ordered_df["Y2H_score"].fillna(0).astype(int).values  # NaN rows excluded via valid_mask

    # ── GCV iterations ────────────────────────────────────────────────────────
    macro_aucs: list[np.ndarray] = []
    micro_aucs: list[list]       = []
    detailed_results             = {"iterations": {}}

    for gcv_seed in range(args.n_gcv):
        print(f"\n{'='*60}", flush=True)
        print(f"GCV seed {gcv_seed}/{args.n_gcv - 1}", flush=True)

        with open(_CV_DIR / cfg.fold_splits_pat.format(seed=gcv_seed), "rb") as f:
            fold_splits = pickle.load(f)
        fold_n_test = [len(test_idx) for _, _, test_idx in fold_splits]

        pair_test_classes = np.load(
            str(_CV_DIR / cfg.pair_test_classes_pat.format(seed=gcv_seed))
        )

        all_preds:  list[float] = []
        all_labels: list[int]   = []

        for fold, train_idx, test_idx in fold_splits:
            # Filter NaN rows (unmatched vt_ids) from train; NaN → NaN pred for test
            train_slice = ordered_df.iloc[train_idx]
            train_slice = train_slice[train_slice["Target_Seq"].notna()].reset_index(drop=True)
            train_df    = swing_to_esignet(train_slice)

            test_slice  = ordered_df.iloc[test_idx].reset_index(drop=True)
            test_valid  = test_slice["Target_Seq"].notna()
            test_df     = swing_to_esignet(test_slice[test_valid].reset_index(drop=True))

            print(
                f"\nFold {fold}: {len(train_slice)} train / {test_valid.sum()} test — "
                f"training ESigNet...",
                flush=True,
            )

            predictor = ESigNetPredictor(
                seed=args.seed,
                n_epochs=args.n_epochs,
                device=args.device,
            )
            predictor.fit(train_df)
            pred_valid = predictor.predict(test_df)  # P(perturbed=1)

            pred_proba = np.full(len(test_idx), np.nan)
            pred_proba[test_valid.values] = pred_valid

            all_preds.extend(pred_proba.tolist())
            all_labels.extend(labels[test_idx].tolist())

            print(f"  Fold {fold} done", flush=True)

        all_preds_arr  = np.array(all_preds)
        all_labels_arr = np.array(all_labels)

        print(f"\nGCV seed {gcv_seed} — per-class AUROCs:", flush=True)
        micro_auc, macro_auc, fold_results = _compute_class_aucs(
            all_preds_arr, all_labels_arr, pair_test_classes, fold_n_test
        )
        print(f"micro AUC (c1/c2/c3): {micro_auc}", flush=True)
        print(f"macro AUC (c1/c2/c3): {macro_auc}", flush=True)

        micro_aucs.append(micro_auc)
        macro_aucs.append(macro_auc)
        detailed_results["iterations"][gcv_seed] = {
            "folds":     fold_results,
            "micro_auc": micro_auc,
            "macro_auc": macro_auc,
        }

    # ── save results ──────────────────────────────────────────────────────────
    stem = f"ESigNet_{cfg.name}"
    np.save(outdir / f"{stem}_micro_aucs.npy", np.array(micro_aucs))
    np.save(outdir / f"{stem}_macro_aucs.npy", np.array(macro_aucs))
    with open(outdir / f"{stem}_detailed_results.pkl", "wb") as f:
        pickle.dump(detailed_results, f)

    print(f"\nResults saved to {outdir}/", flush=True)
    print(f"  {stem}_micro_aucs.npy  shape={np.array(micro_aucs).shape}", flush=True)
    print(f"  {stem}_macro_aucs.npy  shape={np.array(macro_aucs).shape}", flush=True)
    print(f"  {stem}_detailed_results.pkl", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="eSIG-Net GCV training (mirrors SWING pipeline)"
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_CONFIGS),
        help="Dataset configuration to use",
    )
    p.add_argument(
        "--data-root",
        default="/data/ross/ppi_lossgain/interaction_loss/home/data_interaction_loss",
        help="Root directory containing pos/neg/extra CSV files",
    )
    p.add_argument(
        "--device",
        default="",
        help="PyTorch device string (e.g. 'cuda:0'). Defaults to auto-detect.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for eSIG-Net training (default: 42)",
    )
    p.add_argument(
        "--n-gcv",
        type=int,
        default=30,
        help="Number of GCV iterations (default: 30)",
    )
    p.add_argument(
        "--n-epochs",
        type=int,
        default=8,
        help="eSIG-Net training epochs per fold (default: 8)",
    )
    p.add_argument(
        "--outdir",
        default=str(_CV_DIR),
        help="Output directory for results (default: CV splits dir)",
    )
    p.add_argument(
        "--esm-cache",
        default="",
        help="Override ESM-2 residue embeddings cache path (default: built-in path in esignet.py)",
    )
    p.add_argument(
        "--min-esm-hit-rate",
        type=float,
        default=0.9,
        help="Minimum mutant-key hit rate in the ESM cache before "
             "--require-esm aborts the run (default: 0.9).",
    )
    p.add_argument(
        "--require-esm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort if the ESM cache is missing or has hit rate below "
             "--min-esm-hit-rate (default: True). Pass --no-require-esm to "
             "run with the discriminator branch zeroed out.",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
