"""Shared base class for cache-backed MLP predictors.

All embedding-based predictors follow the same pattern:
  1. Load a precomputed embedding cache (lazy, module-level singleton)
  2. In fit(): extract features for each training row, fit sklearn MLP
  3. In predict(): extract features for test rows, return scores

Architecture (following the twin network paper):
  g(z) = h1(mish(h0(z)))
  h0: R^d → R^hidden
  h1: R^hidden → R^out
  classifier: R^out → (0, 1)

Two feature variants per embedding type:
  seq_diff:  mean_pool(emb_mut_a) - mean_pool(emb_wt_a) | mean_pool(emb_b)
  site_diff: emb_mut_a[mut_pos]  - emb_wt_a[mut_pos]   | mean_pool(emb_b)
"""

from __future__ import annotations

import logging
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from ids import parse_mutation  # noqa: E402,F401
# Re-exported: predictors/{mint,pplm}_mlp.py import this relatively from here.

from . import BasePredictor

logger = logging.getLogger(__name__)

# ── Module-level cache singletons (one per path, shared across all folds) ────
_CACHE_STORE: Dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()


def load_cache(path: str) -> Optional[Any]:
    """Load a pickle cache, caching it in memory after the first load.

    Double-checked locking prevents multiple threads from loading the same
    large cache file simultaneously when many fold tasks start at once.
    """
    if path in _CACHE_STORE:  # fast path — no lock after first load
        return _CACHE_STORE[path]
    with _CACHE_LOCK:
        if path in _CACHE_STORE:  # re-check under lock
            return _CACHE_STORE[path]
        p = Path(path)
        if not p.exists():
            logger.warning("Cache not found: %s", path)
            _CACHE_STORE[path] = None
            return None
        logger.info("Loading cache: %s", path)
        with open(p, "rb") as f:
            obj = pickle.load(f)
        _CACHE_STORE[path] = obj
        logger.info("  Loaded %d entries", len(obj) if hasattr(obj, "__len__") else -1)
        return obj


# ── Mutation parsing helpers ──────────────────────────────────────────────────



# ── Feature extraction utilities ─────────────────────────────────────────────

def safe_mean(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None or arr.size == 0:
        return None
    return arr.astype(np.float32).mean(axis=0)


def build_feature(diff: Optional[np.ndarray],
                  partner: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Concatenate diff and partner vectors into a flat feature."""
    if diff is None or partner is None:
        return None
    return np.concatenate([diff.astype(np.float32), partner.astype(np.float32)])


# ── MLP predictor base ───────────────────────────────────────────────────────

class CacheMLPPredictor(BasePredictor):
    """Base class: subclasses must implement _extract_features(row) → ndarray|None.

    PCA variants: set _n_pca (int) on a subclass to fit PCA on training features
    before the StandardScaler/MLP.  This reduces dimensionality and can improve
    generalization to unseen proteins/variants.  None (default) = no PCA.
    """

    # Set in subclass
    _cache_path: str = ""
    _feature_dim: int = 0
    _n_pca: Optional[int] = None  # override in subclass for PCA variants

    @property
    def cache_paths(self) -> list:
        return [self._cache_path] if self._cache_path else []

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._scaler = None
        self._clf = None
        self._pca = None
        self._pre_pca_scaler = None

    def _get_cache(self) -> Optional[Any]:
        return load_cache(self._cache_path)

    def _extract_features(self, row: pd.Series) -> Optional[np.ndarray]:
        """Return fixed-size feature vector or None if cache miss."""
        raise NotImplementedError

    def _extract_all(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract features for all rows; returns (X, y, valid_mask)."""
        from sklearn.preprocessing import StandardScaler

        feats, ys, valid = [], [], []
        for _, row in df.iterrows():
            f = self._extract_features(row)
            if f is not None and np.isfinite(f).all():
                feats.append(f)
                ys.append(int(row["perturbed"]))
                valid.append(True)
            else:
                valid.append(False)

        if not feats:
            return np.empty((0, 0)), np.empty(0), np.array(valid)

        return np.vstack(feats), np.array(ys), np.array(valid)

    def fit(self, train_df: pd.DataFrame) -> None:
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        X, y, mask = self._extract_all(train_df)

        if len(X) < 10:
            logger.warning("%s: only %d valid training samples — using prior.",
                           self.name, len(X))
            self._prior = float(train_df["perturbed"].mean())
            self._clf = None
            self._scaler = None
            self._pca = None
            self._pre_pca_scaler = None
            return

        # Optional PCA dimensionality reduction before scaling.
        # Pre-normalize first: embedding differences may not be unit-scale,
        # so without this, high-variance dimensions dominate the PCA.
        if self._n_pca is not None:
            from sklearn.decomposition import PCA
            self._pre_pca_scaler = StandardScaler()
            X = self._pre_pca_scaler.fit_transform(X)
            n_components = min(self._n_pca, X.shape[0] - 1, X.shape[1])
            self._pca = PCA(n_components=n_components, random_state=self._seed)
            X = self._pca.fit_transform(X)
            logger.debug("%s: PCA %d→%d (%.1f%% variance explained)",
                         self.name, X.shape[1] + n_components, n_components,
                         100 * self._pca.explained_variance_ratio_.sum())
        else:
            self._pca = None
            self._pre_pca_scaler = None

        self._scaler = StandardScaler()
        X_sc = self._scaler.fit_transform(X)

        # Head matches MINT's own downstream evaluation harness, so the MINT and
        # PPLM baselines are run the way MINT's authors prescribe:
        # mint/downstream/GeneralPPI/finetune_general.py::return_logistic_model.
        # (mutational-ppi/README.md points at that script; NOT oncoPPI/train.py.)
        # Upstream wraps this in GridSearchCV, but its MLP param_grid is a single
        # point, so the values below are the whole grid -- no tuning to replicate.
        # Feature differencing likewise follows upstream: baselines.py::encode_two
        # defaults to how="subtract".
        self._clf = MLPClassifier(
            hidden_layer_sizes=(640,),
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=100,
            random_state=self._seed,
            early_stopping=True,
            validation_fraction=0.1,
            tol=1e-4,
            alpha=1e-4,
        )
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            self._prior = float(y.mean())
            self._clf = None
            self._pca = None
            self._pre_pca_scaler = None
            return

        self._clf.fit(X_sc, y)
        self._prior = float(y.mean())

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        prior = self._prior if hasattr(self, "_prior") else 0.5
        scores = np.full(len(test_df), prior)

        if self._clf is None or self._scaler is None:
            return scores

        X, _, valid = self._extract_all(test_df)
        if X.shape[0] == 0:
            return scores

        if self._pca is not None:
            X = self._pre_pca_scaler.transform(X)
            X = self._pca.transform(X)

        X_sc = self._scaler.transform(X)
        proba = self._clf.predict_proba(X_sc)[:, 1]

        valid_indices = np.where(valid)[0]
        scores[valid_indices] = proba

        return scores
