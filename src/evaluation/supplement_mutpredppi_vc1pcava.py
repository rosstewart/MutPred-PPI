#!/usr/bin/env python
"""MutPred-PPI supplemental blind test on VC1p + CAVA entries.

These entries are in the SFVCFP pkl but were excluded from the main VCFP blind
test because their graphs use gene-name/Entrez IDs rather than UniProt IDs.

Proteins are remapped to UniProt via gene_symbol_to_uniprot.pkl (same mapping
used in load_sahni_fragoza_varchamp1p_cava() for GCV construction). C1/C2/C3
classification uses the remapped UniProt IDs against the SF protein set.

22 entries are shared between VC1p and CAVA — they are deduplicated here.

`run()` is importable (e.g. from src/evaluation/run_vcfp_blind_test.py) and
returns the method description string used for the saved npy files.

Usage:
    conda run -n ppi OPENBLAS_NUM_THREADS=1 python supplement_mutpredppi_vc1pcava.py [--device cuda:1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

_CV_MOD  = Path(__file__).resolve().parent  # src/evaluation

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import REPO_ROOT  # noqa: E402

_INF_MOD = REPO_ROOT / "src" / "inference"
# Import model_loader BEFORE cv.py to prevent src/evaluation/utils/ shadowing.
sys.path.insert(0, str(_INF_MOD))
from utils.model_loader import MutPred_PPI, model_predict  # noqa: E402

sys.path.insert(0, str(_CV_MOD))
from vcfp_common import (  # noqa: E402

    build_sf_proteins, load_vc1pcava_sources, iter_vc1pcava_entries,
    save_vc1pcava_supplement,
)


_PUB = REPO_ROOT
_MODEL_PATH = _PUB / "weights" / "MutPred-PPI_sahni_fragoza.pt"
_SCALER_PATH = _PUB / "weights" / "mutation_diff_scaler.pkl"
_DESCRIPTION = "MutPred-PPI (megascale_all, all-data) (varchamp_full_pooled)"


def load_model(device: torch.device):
    m = MutPred_PPI(input_dim=1024).to(device)
    m.load_state_dict(torch.load(str(_MODEL_PATH), map_location=device, weights_only=True))
    m.eval()
    print(f"Loaded: {_MODEL_PATH}")
    return [m]


def run_inference(sources: list[tuple[dict, dict]], sf_proteins: set, models, scaler, device):
    """Run inference over vc1p and cava sources, dedup on UniProt vt_ids.

    sources: list of (data_dict, gs2u_dict) pairs
    Returns arrays: scores, labels, vt_ids_out (UniProt format), class_labels
    """
    scores, labels, vt_ids_out, classes = [], [], [], []
    n_invalid = 0

    for data, i, wt_id, variant, u1, u2, uniprot_vt_id, label, c in iter_vc1pcava_entries(
        sources, sf_proteins
    ):
        mut_idx = int(variant[1:-1])  # 0-based (vc1pcava convention)

        combined = data["prott5_embeddings"][i]
        em       = data["edge_mats"][i]
        raw_diff = data["mutation_site_diffs"][i]
        diff_scaled = scaler.transform(raw_diff.reshape(1, -1)).squeeze()

        score = model_predict(combined, em, models, mut_idx, diff_scaled, device)
        if score is None:
            n_invalid += 1
            continue

        scores.append(float(score))
        labels.append(label)
        vt_ids_out.append(uniprot_vt_id)
        classes.append(c)

    print(f"  invalid={n_invalid}")
    return (
        np.array(scores, dtype=np.float32),
        np.array(labels, dtype=np.int32),
        np.array(vt_ids_out),
        np.array(classes, dtype=np.int32),
    )


def run(device: str = "cuda:1") -> str:
    """Train/predict MutPred-PPI on VC1p+CAVA entries and save the vc1pcava supplement.

    Returns the method description string (also the key expected by
    merge_vc1pcava_into_main.merge_method() / restratify_vcfp_blind_test.restratify_one_method()).
    """
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")

    print("Building SF protein set…")
    sf_proteins = build_sf_proteins()
    print(f"  {len(sf_proteins)} SF UniProt proteins")

    sources = load_vc1pcava_sources()

    print("Loading model…")
    models = load_model(device_t)

    print("Loading scaler…")
    scaler = joblib.load(str(_SCALER_PATH))

    print("\nRunning inference (UniProt remapping + dedup)…")
    scores, labels, vt_ids_out, classes = run_inference(sources, sf_proteins, models, scaler, device_t)

    print(f"\nTotal unique entries: {len(scores)}")
    return save_vc1pcava_supplement(scores, labels, vt_ids_out, classes, _DESCRIPTION)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    run(device=args.device)


if __name__ == "__main__":
    main()
