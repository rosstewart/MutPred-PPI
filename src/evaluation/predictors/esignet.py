"""eSIG-Net predictor.

Adapts the eSIG-Net architecture (Pan et al.) as a benchmark predictor trained
from scratch on our binary PPI data.  Skips the two expensive pairwise-cluster
LOCO splits.

Model:  SdnnModel — dual-channel 573-dim sequence feature network with an
        ESM-2-powered mutation discriminator.
Score:  P(interaction change) = softmax(discriminator_esm output)[:,1].

573-dim features: AAC (20) + Conjoint Triad (343) + auto-covariance (210).
ESM diff:         |emb_mut[pos0] - emb_wt[pos0]|, falls back to zeros on miss.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import BasePredictor, register
from .nn_base import load_cache, parse_mutation

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
from paths import EXTERNAL_DIR, REVISIONS_DIR  # noqa: E402


logger = logging.getLogger(__name__)

# Upstream eSIG-Net source, imported by file path (see docs/METHOD_PROVENANCE.md).
# Resolved via external/esignet -> the pristine clone; run scripts/link_external.sh.
_SDNN_MODEL_PATH = EXTERNAL_DIR / "esignet" / "backbones" / "sdnn" / "sdnn_model.py"
_ESM_CACHE_PATH = str(REVISIONS_DIR / "esm2_residue_embeddings.pkl")
# Sequence → 573-dim feature vector (None on error).  Shared across all folds.
_FEAT_CACHE: dict[str, Optional[np.ndarray]] = {}

# Detected once from the cache keys: True = 1-based mutation suffix ("E80K"),
# False = 0-based ("E79K").


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_sdnn_class():
    """Import SdnnModel without polluting sys.modules."""
    spec = importlib.util.spec_from_file_location("_esignet_sdnn_model", _SDNN_MODEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SdnnModel


# ── Conjoint Triad (Shen 2007 7-group mapping, 343 features) ─────────────────
# Frequency-normalized (count / total) to match upstream eSIG-Net's shipped
# features; see _conjoint_triad and repro_test/validate_esignet_features.py.

_CT_GROUPS = {
    "g1": "AGV", "g2": "ILFP", "g3": "YMTS",
    "g4": "HNQW", "g5": "RK",  "g6": "DE", "g7": "C",
}
_CT_AA_TO_GROUP: dict[str, str] = {
    aa: g for g, aas in _CT_GROUPS.items() for aa in aas
}
_CT_GROUP_KEYS = sorted(_CT_GROUPS)  # ['g1', ..., 'g7']
_CT_FEATURES: list[str] = [
    f"{a}.{b}.{c}"
    for a in _CT_GROUP_KEYS
    for b in _CT_GROUP_KEYS
    for c in _CT_GROUP_KEYS
]  # 343 features in deterministic order


def _conjoint_triad(seq: str) -> np.ndarray:
    """Shen 2007 conjoint triad, 7 groups, gap=0, frequency-normalized.

    Normalization is `count / total`, matching upstream eSIG-Net's shipped
    features (`datasets/embeddings/sdnn_corrected_ppi.h5`): rowsum 1.0000,
    max 0.10256, frac_zero 0.495.  This block previously used LightGBM-PPI's
    `(count - min) / max`, which gives rowsum 44.5 and max 1.0 -- i.e. the CT
    block (343 of the 573 features) entered the SDNN ~45x larger than upstream's
    while AAC and AC matched, distorting the block balance the model sees.
    Verified by `repro_test/validate_esignet_features.py`.
    """
    counts = {f: 0 for f in _CT_FEATURES}
    L = len(seq)
    for i in range(L):
        if i + 1 < L and i + 2 < L:
            try:
                key = (
                    _CT_AA_TO_GROUP[seq[i]] + "."
                    + _CT_AA_TO_GROUP[seq[i + 1]] + "."
                    + _CT_AA_TO_GROUP[seq[i + 2]]
                )
            except KeyError:
                continue  # non-canonical residue: skip
            counts[key] += 1
    arr = np.array([counts[f] for f in _CT_FEATURES], dtype=np.float32)
    total = float(arr.sum())
    if total == 0:
        return arr
    return arr / total


# ── Auto-covariance (verbatim port of SDNN-PPI/Feature Extraction/ac.py) ──────
# Y has NCI=117.3, V=0.023599.  These look transposed (every other residue has
# NCI in [0.003, 0.24] and V in [29, 145]) but they are NOT a transcription
# error: reproducing upstream's shipped features requires them exactly as-is.
# Per-property variance profile vs upstream's h5 correlates +0.99 with this
# table and -0.25 with the 'corrected' one.  Do not 'fix' this.
# See repro_test/validate_esignet_features.py.

_AC_PROPERTIES = ["H1", "H2", "NCI", "P1", "P2", "SASA", "V"]
_AC_AA_PROPS: dict[str, dict[str, float]] = {
    "A": {"H1":  0.62, "H2": -0.5, "NCI":  0.007187, "P1":  8.1, "P2": 0.046, "SASA": 1.181, "V":  27.5},
    "C": {"H1":  0.29, "H2": -1.0, "NCI": -0.036610, "P1":  5.5, "P2": 0.128, "SASA": 1.461, "V":  44.6},
    "D": {"H1": -0.90, "H2":  3.0, "NCI": -0.023820, "P1": 13.0, "P2": 0.105, "SASA": 1.587, "V":  40.0},
    "E": {"H1":  0.74, "H2":  3.0, "NCI":  0.006802, "P1": 12.3, "P2": 0.151, "SASA": 1.862, "V":  62.0},
    "F": {"H1":  1.19, "H2": -2.5, "NCI":  0.037552, "P1":  5.2, "P2": 0.290, "SASA": 2.228, "V": 115.5},
    "G": {"H1":  0.48, "H2":  0.0, "NCI":  0.179052, "P1":  9.0, "P2": 0.000, "SASA": 0.881, "V":   0.0},
    "H": {"H1": -0.40, "H2": -0.5, "NCI": -0.010690, "P1": 10.4, "P2": 0.230, "SASA": 2.025, "V":  79.0},
    "I": {"H1":  1.38, "H2": -1.8, "NCI":  0.021631, "P1":  5.2, "P2": 0.186, "SASA": 1.810, "V":  93.5},
    "K": {"H1": -1.50, "H2":  3.0, "NCI":  0.017708, "P1": 11.3, "P2": 0.219, "SASA": 2.258, "V": 100.0},
    "L": {"H1":  1.06, "H2": -1.8, "NCI":  0.051672, "P1":  4.9, "P2": 0.186, "SASA": 1.931, "V":  93.5},
    "M": {"H1":  0.64, "H2": -1.3, "NCI":  0.002683, "P1":  5.7, "P2": 0.221, "SASA": 2.034, "V":  94.1},
    "N": {"H1": -0.78, "H2":  2.0, "NCI":  0.005392, "P1": 11.6, "P2": 0.134, "SASA": 1.655, "V":  58.7},
    "P": {"H1":  0.12, "H2":  0.0, "NCI":  0.239531, "P1":  8.0, "P2": 0.131, "SASA": 1.468, "V":  41.9},
    "Q": {"H1": -0.85, "H2":  0.2, "NCI":  0.049211, "P1": 10.5, "P2": 0.180, "SASA": 1.932, "V":  80.7},
    "R": {"H1": -2.53, "H2":  3.0, "NCI":  0.043587, "P1": 10.5, "P2": 0.291, "SASA": 2.560, "V": 105.0},
    "S": {"H1": -0.18, "H2":  0.3, "NCI":  0.004627, "P1":  9.2, "P2": 0.062, "SASA": 1.298, "V":  29.3},
    "T": {"H1": -0.05, "H2": -0.4, "NCI":  0.003352, "P1":  8.6, "P2": 0.108, "SASA": 1.525, "V":  51.3},
    "V": {"H1":  1.08, "H2": -1.5, "NCI":  0.057004, "P1":  5.9, "P2": 0.140, "SASA": 1.645, "V":  71.5},
    "W": {"H1":  0.81, "H2": -3.4, "NCI":  0.037977, "P1":  5.4, "P2": 0.409, "SASA": 2.663, "V": 145.5},
    "Y": {"H1":  0.26, "H2": -2.3, "NCI":  117.3000, "P1":  6.2, "P2": 0.298, "SASA": 2.368, "V":   0.023599},
}


def _build_ac_norm() -> dict[str, dict[str, float]]:
    """Z-score normalize each property across the 20 AAs (matches ac.py)."""
    norm: dict[str, dict[str, float]] = {aa: {} for aa in _AC_AA_PROPS}
    for prop in _AC_PROPERTIES:
        vals = [_AC_AA_PROPS[aa][prop] for aa in _AC_AA_PROPS]
        avg = sum(vals) / len(vals)
        var = sum((v - avg) ** 2 for v in vals) / len(vals)
        std = var ** 0.5
        for aa in _AC_AA_PROPS:
            norm[aa][prop] = (_AC_AA_PROPS[aa][prop] - avg) / std
    return norm


_AC_NORM = _build_ac_norm()


def _ac_code(seq: str) -> np.ndarray:
    """Auto-covariance: 7 properties × 30 lags = 210, property-major order.

    For each property: encode → per-protein centering → AC(lag) = (1/(L-lag)) Σ Sᵢ Sᵢ₊lag.
    """
    out: list[float] = []
    for prop in _AC_PROPERTIES:
        vals = [_AC_NORM[aa][prop] for aa in seq if aa in _AC_NORM]
        L = len(vals)
        avg = sum(vals) / L if L > 0 else 0.0
        centered = [v - avg for v in vals]
        for lag in range(1, 31):
            if L > lag:
                s = sum(centered[i] * centered[i + lag] for i in range(L - lag))
                out.append(s / (L - lag))
            else:
                out.append(0.0)
    return np.array(out, dtype=np.float32)


def _compute_573(seq: str) -> Optional[np.ndarray]:
    """AAC (20) + Conjoint Triad (343) + Auto-Covariance (210) = 573-dim.

    AAC: protlearn frequency-normalized (paper: "normalized frequency of occurrence").
    CT:  Shen 2007 7-group, gap=0, frequency-normalized to match upstream's h5.
    AC:  7 physicochemical properties × 30 lags — verbatim SDNN-PPI/ac.py.
    """
    if seq in _FEAT_CACHE:
        return _FEAT_CACHE[seq]
    try:
        from protlearn.features import aac  # noqa: PLC0415
        aac_feat = aac(seq)[0].flatten()              # (20,)
        ct_feat  = _conjoint_triad(seq)               # (343,)
        ac_feat  = _ac_code(seq)                      # (210,)
        feat = np.concatenate([aac_feat, ct_feat, ac_feat]).astype(np.float32)
        assert feat.shape[0] == 573, f"unexpected shape {feat.shape}"
        _FEAT_CACHE[seq] = feat
        return feat
    except Exception as exc:
        logger.warning("573-feat failed (len %d): %s", len(seq), exc)
        _FEAT_CACHE[seq] = None
        return None


def _apply_mutation(seq: str, mutation: str) -> str:
    """Apply a point mutation to a sequence string (1-based notation)."""
    _, pos0, mut_aa = parse_mutation(mutation)
    return seq[:pos0] + mut_aa + seq[pos0 + 1:]


def _esm_diff(cache: dict, interactor: str, mutation: str) -> np.ndarray:
    """Absolute ESM-2 embedding diff at mutation site; zeros on cache miss.

    Supports windowed embeddings: if the cache holds a {key}_offset entry the
    mutant embedding was computed on a window starting at that offset, so the
    lookup index is pos0 - offset rather than pos0.
    """
    # Cache keys are 1-based, matching the canonical tables. There is no base
    # detection: a miss is a miss, not a cue to try the other convention.
    _, pos0, _ = parse_mutation(mutation)
    mt_key = f"{interactor}_{mutation}"

    wt = cache.get(interactor)
    mt = cache.get(mt_key)
    if wt is None or mt is None:
        return np.zeros(1280, dtype=np.float32)

    wt_offset: int = cache.get(f"{interactor}_offset", 0)
    mt_offset: int = cache.get(f"{mt_key}_offset", 0)
    wt_idx = pos0 - wt_offset
    mt_idx = pos0 - mt_offset

    if 0 <= wt_idx < wt.shape[0] and 0 <= mt_idx < mt.shape[0]:
        return np.abs(mt[mt_idx].astype(np.float32) - wt[wt_idx].astype(np.float32))

    logger.debug(
        "ESM diff OOB: %s %s pos0=%d wt_idx=%d/%d mt_idx=%d/%d — using zeros",
        interactor, mutation, pos0, wt_idx, wt.shape[0], mt_idx, mt.shape[0],
    )
    return np.zeros(1280, dtype=np.float32)


def _build_tensors(
    df: pd.DataFrame,
    esm_cache: Optional[dict],
) -> tuple:
    """Build eight tensors (n0, n1, y, n0_2, n1_2, y_2, same, esm_feat)."""
    n = len(df)
    n0_arr   = np.zeros((n, 573), dtype=np.float32)
    n1_arr   = np.zeros((n, 573), dtype=np.float32)
    n0_2_arr = np.zeros((n, 573), dtype=np.float32)
    y_arr    = np.ones(n, dtype=np.int64)          # WT always interacts
    y_2_arr  = np.empty(n, dtype=np.int64)
    same_arr = np.zeros(n, dtype=np.float32)       # WT ≠ mutant always
    esm_arr  = np.zeros((n, 1280), dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        seq_wt = row["interactor_sequence"]
        seq_b  = row["partner_sequence"]
        seq_mt = _apply_mutation(seq_wt, row["mutation"])

        f0  = _compute_573(seq_wt)
        f1  = _compute_573(seq_b)
        f02 = _compute_573(seq_mt)
        if f0  is not None: n0_arr[i]   = f0
        if f1  is not None: n1_arr[i]   = f1
        if f02 is not None: n0_2_arr[i] = f02

        y_2_arr[i] = int(1 - row["perturbed"])

        if esm_cache is not None:
            esm_arr[i] = _esm_diff(esm_cache, row["interactor"], row["mutation"])

    return (
        torch.from_numpy(n0_arr),
        torch.from_numpy(n1_arr),
        torch.from_numpy(y_arr),
        torch.from_numpy(n0_2_arr),
        torch.from_numpy(n1_arr.copy()),   # n1_2 == n1 (partner unchanged)
        torch.from_numpy(y_2_arr),
        torch.from_numpy(same_arr),
        torch.from_numpy(esm_arr),
    )


def _pair_loss(merged_wt: torch.Tensor, merged_mt: torch.Tensor,
               y: torch.Tensor, y_2: torch.Tensor,
               max_edit_dist: int) -> torch.Tensor:
    """Intra-batch contrastive pair loss (EditLoss, func='sum'), normalised."""
    dist_norms = []
    for j in range(len(y)):
        edit_dist = 1 if y[j].item() == y_2[j].item() else max_edit_dist
        l2 = nn.functional.pairwise_distance(
            merged_wt[j].unsqueeze(0), merged_mt[j].unsqueeze(0)
        ).squeeze()
        dist_norms.append((l2 / edit_dist).unsqueeze(0))
    dist_tensor = torch.cat(dist_norms, dim=0)    # (B,)

    # EditLoss(func='sum'): sum of pairwise MSEs between normalised distances
    n = dist_tensor.size(0)
    expanded = dist_tensor.unsqueeze(0).expand(n, -1)
    mse_mat  = nn.functional.mse_loss(expanded, expanded.T, reduction="none")
    return mse_mat.sum() / max(n, 1)


# ── predictor ─────────────────────────────────────────────────────────────────

@register
class ESigNetPredictor(BasePredictor):
    """eSIG-Net: 573-dim dual-channel SDNN + ESM-2 discriminator.

    Trained from scratch on labelled PPI data.
    Skipped for expensive pairwise-cluster LOCO splits.
    """

    skip_scenarios = frozenset({"cluster_loco_pairwise", "edgetic_cluster_loco_pairwise"})
    cache_paths = [_ESM_CACHE_PATH]
    preferred_n_jobs = 0  # Use global --n-jobs; requires --backend loky for GPU

    def __init__(
        self,
        seed: int = 42,
        n_epochs: int = 8,
        lr: float = 1e-3,
        batch_size: int = 32,
        pair_weight: float = 0.05,
        discrim_weight: float = 1.0,
        max_edit_dist: int = 2,
        dropout_p: float = 0.3,
        device: str = "",
    ):
        self._seed = seed
        self._n_epochs = n_epochs
        self._lr = lr
        self._batch_size = batch_size
        self._pair_weight = pair_weight
        self._discrim_weight = discrim_weight
        self._max_edit_dist = max_edit_dist
        self._dropout_p = dropout_p
        self._device_str = device
        self._model: Optional[nn.Module] = None
        self._device: Optional[torch.device] = None

    @property
    def name(self) -> str:
        return "esignet"

    def _device_of(self) -> torch.device:
        if self._device is None:
            if self._device_str:
                self._device = torch.device(self._device_str)
            else:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    def fit(self, train_df: pd.DataFrame) -> None:
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)

        device = self._device_of()
        esm_cache = load_cache(_ESM_CACHE_PATH)

        logger.info("ESigNet: building features for %d training rows", len(train_df))
        n0, n1, y, n0_2, n1_2, y_2, same, esm_feat = _build_tensors(
            train_df, esm_cache
        )

        loader = DataLoader(
            TensorDataset(n0, n1, y, n0_2, n1_2, y_2, same, esm_feat),
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True,    # BatchNorm1d needs ≥2 samples
        )

        SdnnModel = _load_sdnn_class()
        model = SdnnModel(
            in_features=573, out_features=32,
            dropout_p=self._dropout_p, use_esm=True,
        ).to(device)
        model.train()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self._lr, weight_decay=1e-4
        )
        ce_fn = nn.CrossEntropyLoss()

        for epoch in range(self._n_epochs):
            total = 0.0
            steps = 0
            for batch in loader:
                b_n0, b_n1, b_y, b_n02, b_n12, b_y2, _, b_esm = [
                    t.to(device) for t in batch
                ]

                pred,  m_wt = model(b_n0, b_n1)
                pred_2, m_mt = model(b_n02, b_n12)
                pred_change, _, _ = model.discriminator_esm(m_wt, m_mt, b_esm)

                bce      = ce_fn(pred, b_y.long())
                p_loss   = _pair_loss(m_wt, m_mt, b_y, b_y2, self._max_edit_dist)
                d_loss   = ce_fn(pred_change, (b_y != b_y2).long())

                loss = bce + self._pair_weight * p_loss + self._discrim_weight * d_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total += loss.item()
                steps += 1

            logger.info(
                "ESigNet epoch %d/%d  loss=%.4f",
                epoch + 1, self._n_epochs, total / max(steps, 1),
            )

        model.eval()
        self._model = model

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            logger.warning("ESigNet.predict called before fit(); returning 0.5")
            return np.full(len(test_df), 0.5, dtype=np.float32)

        device = self._device_of()
        esm_cache = load_cache(_ESM_CACHE_PATH)

        n0, n1, _, n0_2, n1_2, _, _, esm_feat = _build_tensors(
            test_df, esm_cache
        )

        scores: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(n0), self._batch_size):
                end = min(start + self._batch_size, len(n0))
                b_n0  = n0[start:end].to(device)
                b_n1  = n1[start:end].to(device)
                b_n02 = n0_2[start:end].to(device)
                b_n12 = n1_2[start:end].to(device)
                b_esm = esm_feat[start:end].to(device)

                _, m_wt = self._model(b_n0, b_n1)
                _, m_mt = self._model(b_n02, b_n12)
                pred_change, _, _ = self._model.discriminator_esm(m_wt, m_mt, b_esm)
                probs = torch.softmax(pred_change, dim=1)[:, 1]
                scores.append(probs.cpu().numpy())

        return np.concatenate(scores).astype(np.float32)
