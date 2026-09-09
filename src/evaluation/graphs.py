#!/usr/bin/env python
"""Resolve a protein pair to its AF3 contact graph, by sequence.

Structures are content-addressed: the key is the pair's two chain sequences, not
the filename. Accessions change between mappings while sequences do not, so an
ID-keyed lookup discards valid structures and can accept one whose name matches
but whose sequence does not.

    resolver = GraphResolver()
    G, ok = resolver.load(interactor_sequence, partner_sequence)

`G` is the contact matrix oriented so the interactor is chain A. A pair with no
exact sequence match returns (None, False) -- there is no fuzzy fallback.

The index is built by repro_test/build_af3_index.py.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import loadmat

from paths import DATA_ROOT, DATASETS_DIR, REVISIONS_DIR  # noqa: E402

INDEX_PATH = DATASETS_DIR / "mapped090826" / "af3_index.csv.gz"

# Searched in order; the first directory holding a given structure wins.
PDB_DIRS = [
    DATA_ROOT / "three_datasets_af3_models" / "pdbs",
    DATA_ROOT / "sahni_pdbs",
    REVISIONS_DIR / "af3_out" / "pdbs",
    REVISIONS_DIR / "af3_out" / "varchamp_pooled" / "pdbs",
    DATA_ROOT / "out_sfvc_5seed" / "pdbs",
]


def sha(seq: str) -> str:
    return hashlib.sha256(str(seq).encode()).hexdigest()[:16]


def orient(G: np.ndarray, nrr: int, reversed_graph: bool):
    """Rotate a reversed graph so the interactor occupies chain A.

    When the stored graph has the partner first, both the adjacency matrix and
    any concatenated sequence must be rotated by `nrr`, or the mutation index
    addresses the wrong chain.
    """
    if not reversed_graph:
        return G
    n = G.shape[0]
    perm = list(range(nrr, n)) + list(range(nrr))
    return G[np.ix_(perm, perm)]


class GraphResolver:
    def __init__(self, index_path: Path | str = INDEX_PATH):
        self._idx = {}
        p = Path(index_path)
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found -- run repro_test/build_af3_index.py first")
        df = pd.read_csv(p)
        for r in df.itertuples(index=False):
            self._idx[(r.seq_a_sha, r.seq_b_sha)] = (r.path, bool(r.chain_a_is_first),
                                                     int(r.len_a))
        self.n_indexed = len(self._idx)

    def find(self, seq_a: str, seq_b: str):
        """(path, interactor_is_chain_a, chain_a_len) or None. Order-independent."""
        ha, hb = sha(seq_a), sha(seq_b)
        hit = self._idx.get((min(ha, hb), max(ha, hb)))
        if hit is None:
            return None
        path, a_is_first, len_a = hit
        # `a_is_first` describes the stored file in sorted-hash terms; translate
        # that into "is our interactor chain A".
        interactor_is_first = (ha <= hb) == a_is_first
        return path, interactor_is_first, len_a

    def load(self, seq_a: str, seq_b: str):
        """Contact matrix oriented interactor-first, or (None, False) on a miss."""
        hit = self.find(seq_a, seq_b)
        if hit is None:
            return None, False
        path, interactor_is_first, _ = hit
        mat = loadmat(path)
        G = sp.csr_matrix(mat["G"]).toarray()
        nrr = int(np.asarray(mat["NRR"]).ravel()[0])
        stored = "".join(str(x) for x in np.asarray(mat["L"]).ravel())
        chain_a, chain_b = stored[:nrr], stored[nrr:]
        # Assert at load, not just at index time, so a mislabelled or swapped
        # file cannot slip through.
        want = {sha(seq_a), sha(seq_b)}
        if {sha(chain_a), sha(chain_b)} != want:
            raise ValueError(f"{path}: chain sequences do not match the requested pair")
        return orient(G, nrr, reversed_graph=not interactor_is_first), True

    def coverage(self, rows: pd.DataFrame):
        """(n_found, n_missing) for a canonical rows frame."""
        pairs = rows.drop_duplicates(["interactor", "partner"])
        found = sum(1 for a, b in zip(pairs["interactor_sequence"],
                                      pairs["partner_sequence"])
                    if self.find(a, b) is not None)
        return found, len(pairs) - found


def _pdb_key(stem: str) -> str:
    """Normalise a .mat or .pdb stem to a name-convention-independent key.

    The structure trees use two incompatible conventions -- `fold_a_b_model_0.pdb`
    in the older dirs and `a-b_model.pdb` in the AF3 output dirs -- and the .mat
    stems use hyphens or underscores as the pair separator depending on which
    pipeline wrote them. Normalising both sides identically matches across all of
    them; matching on one convention alone silently finds only a quarter of the
    structures that exist.
    """
    s = stem.lower()
    if s.startswith("fold_"):
        s = s[5:]
    s = re.sub(r"_model(_\d+)?$", "", s)
    return s.replace("-", "_")


class PDBResolver:
    """Locate the PDB file backing a contact graph, for the structure-based methods.

    SAAMBE-3D and DDMut-PPI score a PDB rather than a contact matrix, so they pair
    `GraphResolver.find` (which validates the pair by sequence) with this, which
    maps the resulting .mat path to its PDB. Coverage is partial: many pairs kept
    a contact graph while the original structure was not retained.
    """

    def __init__(self, pdb_dirs=None):
        self._idx: dict[str, Path] = {}
        for d in (PDB_DIRS if pdb_dirs is None else pdb_dirs):
            d = Path(d)
            if not d.is_dir():
                continue
            for p in d.glob("*.pdb"):
                self._idx.setdefault(_pdb_key(p.stem), p)
        self.n_indexed = len(self._idx)

    def find(self, mat_path: str | Path) -> Path | None:
        """PDB backing this .mat file, or None when it was not retained."""
        return self._idx.get(_pdb_key(Path(mat_path).stem))
