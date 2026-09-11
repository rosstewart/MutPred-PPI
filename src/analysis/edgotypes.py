"""Edgotype definition and the on-disk contract for variant-database groups.

One tidy table per variant group, one row per scored variant-partner pair:

    uniprot  variant  partner  score  n_biogrid_partners

Everything the figures need is derived from that table on read -- the edgotype
of a variant, the per-variant score lists the partner-controlled analysis
resamples, and the coverage evidence behind the manuscript's partner rule.

This replaces a pair of parallel files (`{name}_edgotype_classes.npy` and
`{name}_posterior_ls.pkl`) that had to be kept index-aligned by hand, with
nothing checking that they were. Two further problems went with them: the
edgotype array baked in a disruption threshold, so it silently disagreed with
any analysis that swept the threshold; and neither file recorded which variant a
row belonged to, so a result could not be joined back to its variant, audited,
or debugged.

`n_biogrid_partners` is the number of partners BioGRID lists for the variant
within its group, which is not the number scored -- a variant is only analysed
when enough of its known partners could actually be scored (see
`classify_variant_dbs.build_arrays`). Persisting it keeps that decision
reproducible from the outputs alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Ordered as the figures stack them.
EDGOTYPES: tuple[str, ...] = ("Quasi-wild-type", "Quasi-null", "Edgetic")

COLUMNS: list[str] = ["uniprot", "variant", "partner", "score", "n_biogrid_partners"]

DEFAULT_THRESHOLD = 0.5


def classify(scores, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Edgotype of one variant from its partner scores.

    A partner is perturbed when its score is strictly above `threshold`. All
    perturbed is Quasi-null, none perturbed is Quasi-wild-type, a mix is
    Edgetic -- so a variant scored against a single partner can never be
    Edgetic, which is a property of the coverage, not of the variant.
    """
    scores = list(scores)
    n_disrupted = sum(s > threshold for s in scores)
    if n_disrupted == len(scores):
        return "Quasi-null"
    if n_disrupted == 0:
        return "Quasi-wild-type"
    return "Edgetic"


@dataclass(frozen=True)
class EdgotypeGroup:
    """One variant group: the tidy table plus the views the figures want."""

    name: str
    table: pd.DataFrame

    def __len__(self) -> int:
        return len(self.variants)

    @property
    def variants(self) -> list[tuple[str, str]]:
        return list(self._grouped().groups)

    def _grouped(self):
        # sort=True gives a deterministic variant order, so every derived view
        # below indexes the same variant at the same position.
        return self.table.groupby(["uniprot", "variant"], sort=True)

    def scores_by_variant(self) -> list[list[float]]:
        """Per-variant partner scores, in `variants` order."""
        return [g["score"].tolist() for _, g in self._grouped()]

    def classify(self, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
        """Edgotype per variant, in `variants` order."""
        return np.array([classify(s, threshold) for s in self.scores_by_variant()],
                        dtype=object)

    def counts(self, threshold: float = DEFAULT_THRESHOLD) -> dict[str, int]:
        cls = self.classify(threshold)
        return {e: int(np.sum(cls == e)) for e in EDGOTYPES}

    def n_partners(self) -> np.ndarray:
        """Scored partners per variant, in `variants` order."""
        return self._grouped().size().to_numpy()


def group_path(data_dir: str | Path, db: str, name: str) -> Path:
    return Path(data_dir) / db / f"{name}.csv.gz"


def save_group(out_dir: str | Path, name: str, table: pd.DataFrame) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    missing = [c for c in COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"{name}: table is missing {missing}")
    path = out / f"{name}.csv.gz"
    table[COLUMNS].to_csv(path, index=False, compression="gzip")
    return path


def load_group(data_dir: str | Path, db: str, name: str) -> EdgotypeGroup | None:
    """Load one group, or None when it was never produced."""
    path = group_path(data_dir, db, name)
    if not path.exists():
        return None
    return EdgotypeGroup(name=name, table=pd.read_csv(path))
