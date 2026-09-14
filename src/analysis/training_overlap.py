#!/usr/bin/env python3
"""Variants the model was trained on, and how to exclude them.

MutPred-PPI's variant-repository analyses ask whether disruptive predictions are
*enriched* in one class of variants over another. A variant that appears in the
training set has a fitted answer rather than a predicted one, so leaving it in
lets training signal leak into the enrichment and inflates exactly the contrast
the figures are measuring.

Overlap is defined on the **(interactor, variant) pair, ignoring the partner**.
That is deliberately stricter than matching the full (interactor, partner,
variant) triple: a variant seen against *any* partner during training has had its
mutation representation fitted, and the GAT reads the same mutated-site features
whichever partner it is paired with. The triple-level rule would keep those.

Positions are 1-based on both sides -- training tables store `mutation`, variant
repositories store `variant`, both `{WT}{pos1}{MUT}` -- so no base conversion is
involved. `tests/test_training_overlap.py` pins that.

The same definition is used to FILTER the figure inputs and to ANNOTATE the
deposited master tables (`training_overlap` column), so a reader can reproduce
either view from the deposit.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from utils.gcv_common import dataset_config, dataset_name, load_data

# The union of everything any released model was fine-tuned on. The all-data
# model (weights/MutPred-PPI.pt) that scores every variant repository is trained
# on exactly this set, so it is the right thing to exclude.
TRAINING_DATASET = "sahni_fragoza_varchamp_all"

#: Column name added to the deposited master tables.
OVERLAP_COLUMN = "training_overlap"


@lru_cache(maxsize=4)
def training_variants(dataset: str = TRAINING_DATASET) -> frozenset[tuple[str, str]]:
    """{(interactor, variant)} seen in training, partner ignored.

    Raises if the training table is absent: silently returning an empty set
    would turn "no overlap data" into "no overlap", which is the failure mode
    this module exists to prevent.
    """
    rows = load_data(dataset_config(dataset_name(dataset)))
    return frozenset(zip(rows["interactor"].astype(str),
                         rows["mutation"].astype(str)))


def training_variants_or_none(dataset: str = TRAINING_DATASET):
    """As `training_variants`, but None when the training table is unavailable.

    The training set contains unpublished VarChAMP measurements and is not in
    the public deposit. Callers that must degrade gracefully use this and say
    so in their output, rather than silently reporting unfiltered numbers as
    filtered ones.
    """
    try:
        return training_variants(dataset)
    except Exception:                                          # noqa: BLE001
        return None


def overlap_mask(df: pd.DataFrame, uniprot_col: str = "uniprot",
                 variant_col: str = "variant",
                 dataset: str = TRAINING_DATASET) -> pd.Series:
    """Boolean Series: True where (uniprot, variant) was in the training set."""
    known = training_variants(dataset)
    keys = zip(df[uniprot_col].astype(str), df[variant_col].astype(str))
    return pd.Series([k in known for k in keys], index=df.index, dtype=bool)


def drop_training_variants(df: pd.DataFrame, uniprot_col: str = "uniprot",
                           variant_col: str = "variant",
                           dataset: str = TRAINING_DATASET,
                           label: str = "") -> tuple[pd.DataFrame, int]:
    """Remove rows whose (uniprot, variant) appears in training.

    Returns the filtered frame and the number of ROWS removed. Prints a one-line
    report so a run's logs record how much was excluded -- an unexplained change
    in a published number should always be traceable to a line like this.
    """
    if df is None or df.empty:
        return df, 0
    mask = overlap_mask(df, uniprot_col, variant_col, dataset)
    n = int(mask.sum())
    if n:
        where = f" [{label}]" if label else ""
        print(f"  training-overlap filter{where}: dropped {n:,} of {len(df):,} rows "
              f"({100 * n / len(df):.2f}%)", flush=True)
    return df.loc[~mask].reset_index(drop=True), n
