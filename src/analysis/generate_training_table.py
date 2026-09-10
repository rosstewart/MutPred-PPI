#!/usr/bin/env python3
"""Generate the LaTeX training-data table for the MutPred-PPI paper.

Every count comes from the CANONICAL row tables (`datasets/mapped090826/`) via
`utils.gcv_common.load_data`, one table per row of the figure. Those tables are
the same rows the models train on, so the table cannot drift from the training
set the way it used to.

What this replaces, and why it was wrong
----------------------------------------
The previous version read three different `*_all_vt_ids_and_labels.txt` files
and a `*_all_vt_ids.pkl`, and had to recover `(interactor, partner)` from a
welded `complex_id` by trying `_` and then `-`. That heuristic silently split
`NP_005190_KRTAP10-7` and every isoform accession in the wrong place, so the
protein/pair counts it produced were not the counts of anything. It then
*subtracted overlapping label files from each other* to get a VarChAMP row, and
because the files overlapped it had to correct the result with an estimated
disruption rate (`pool_rate`) applied to a row-count difference. None of that
survives: the canonical tables are deduplicated, carry `perturbed` as a real
label, and exist for each subset the table needs.

Consequence: the numbers change. The old SF row reported 5,894 triplets (a stale
ordering); the canonical Sahni+Fragoza table has 6,219.

Writes figures/training_data_table.tex as a drop-in tabular block.
"""
from __future__ import annotations

import pickle

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
from paths import DATA_ROOT, REPO_ROOT  # noqa: E402
from utils.gcv_common import DATASET_CONFIGS, load_data  # noqa: E402
from utils.identifiers import split_variant_id  # noqa: E402


_PUB = REPO_ROOT
_MS = DATA_ROOT / "megascale_preprocessed"
_OUT = _PUB / "figures" / "training_data_table.tex"

# Table row -> canonical dataset. Each is a real table, so no row of the figure
# is derived by subtracting one file from another any more.
_SAHNI = "sahni_only_mapped090826"
_FRAGOZA = "fragoza_only_mapped090826"
_VARCHAMP = "varchamp_all_mapped090826"
_SF = "sahni_fragoza_mapped090826"
_SFVC = "sahni_fragoza_varchamp_all_mapped090826"


def fmt(n) -> str:
    if n is None:
        return r"-"
    return f"{int(n):,}".replace(",", "{,}")


def stats(dataset: str) -> dict:
    """Protein / pair / variant / triplet counts for one canonical table.

    `interactor` and `partner` are columns, so a pair is a tuple of two fields
    rather than a string that has to be taken apart again later. Accessions are
    used exactly as the table stores them -- collapsing an isoform onto its
    parent here would merge two rows the model treats as distinct.
    """
    df = load_data(DATASET_CONFIGS[dataset])
    pairs = set(zip(df["interactor"], df["partner"]))
    variants = set(zip(df["interactor"], df["mutation"]))
    n_dis = int((df["perturbed"] == 1).sum())
    out = dict(
        proteins=len(set(df["interactor"]) | set(df["partner"])),
        pairs=len(pairs),
        variants=len(variants),
        triplets=len(df),
        dis=n_dis,
        non=len(df) - n_dis,
    )
    print(f"{dataset}: {out}", flush=True)
    return out


def megascale_stats() -> dict:
    """Tsuboyama stability pretraining set: proteins, rows, and ddG sign split.

    `vt_ids` here are `'{construct} {mutation}'`, the space-delimited variant id
    of `utils.identifiers`, so the protein side is read with `split_variant_id`
    rather than an ad-hoc `split(" ")[0]`.
    """
    ms = pickle.loads((_MS / "preprocessed.pkl").read_bytes())
    vt_ids, ddg = ms["vt_ids"], ms["ddg_labels"]
    out = dict(
        proteins=len({split_variant_id(vt)[0] for vt in vt_ids}),
        total=len(vt_ids),
        dis=int((ddg < 0).sum()),
        non=int((ddg >= 0).sum()),
    )
    print(f"megascale: {out}", flush=True)
    return out


def main() -> None:
    sahni = stats(_SAHNI)
    fragoza = stats(_FRAGOZA)
    varchamp = stats(_VARCHAMP)
    sf = stats(_SF)
    sfvc = stats(_SFVC)
    ms = megascale_stats()

    # The combined tables are smaller than the sum of their parts: a triple that
    # appears in two sources is one row, not two. Reported so the table is not
    # read as an arithmetic error.
    print(f"\nnote: {_SF} has {sf['triplets']} rows vs "
          f"{sahni['triplets'] + fragoza['triplets']} summed over its two "
          f"sources; the difference is deduplicated overlap.", flush=True)

    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        (r"\textbf{Dataset} & \textbf{Proteins} & \textbf{Pairs} & "
         r"\textbf{Variants} & \textbf{Triplets} & \textbf{Disruptive} & "
         r"\textbf{Non-disruptive} \\"),
        r"\hline",
        r"\multicolumn{7}{l}{\textit{PPI Perturbation Data}} \\",
        (rf"Sahni \textit{{et al.}}~(Mendelian) & {fmt(sahni['proteins'])} & "
         rf"{fmt(sahni['pairs'])} & {fmt(sahni['variants'])} & "
         rf"{fmt(sahni['triplets'])} & {fmt(sahni['dis'])} & {fmt(sahni['non'])} \\"),
        (rf"Fragoza \textit{{et al.}}~(Population) & {fmt(fragoza['proteins'])} & "
         rf"{fmt(fragoza['pairs'])} & {fmt(fragoza['variants'])} & "
         rf"{fmt(fragoza['triplets'])} & {fmt(fragoza['dis'])} & {fmt(fragoza['non'])} \\"),
        (rf"VarChAMP (IGVF) & {fmt(varchamp['proteins'])} & {fmt(varchamp['pairs'])} & "
         rf"{fmt(varchamp['variants'])} & {fmt(varchamp['triplets'])} & "
         rf"{fmt(varchamp['dis'])} & {fmt(varchamp['non'])} \\"),
        r"\hline",
        r"\multicolumn{7}{l}{\textit{Combined Training Sets}} \\",
        (rf"Sahni, Fragoza & {fmt(sf['proteins'])} & {fmt(sf['pairs'])} & "
         rf"{fmt(sf['variants'])} & {fmt(sf['triplets'])} & "
         rf"{fmt(sf['dis'])} & {fmt(sf['non'])} \\"),
        (rf"Sahni, Fragoza, VarChAMP & {fmt(sfvc['proteins'])} & "
         rf"{fmt(sfvc['pairs'])} & {fmt(sfvc['variants'])} & "
         rf"{fmt(sfvc['triplets'])} & {fmt(sfvc['dis'])} & "
         rf"{fmt(sfvc['non'])} \\"),
        r"\hline",
        r"\multicolumn{7}{l}{\textit{Stability Pretraining Data}} \\",
        (rf"Tsuboyama \textit{{et al.}} & {fmt(ms['proteins'])} & - & "
         rf"{fmt(ms['total'])} & - & {fmt(ms['dis'])} & {fmt(ms['non'])} \\"),
        r"\hline",
        r"\end{tabular}",
    ]

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text("\n".join(lines) + "\n")
    print(f"\nWrote → {_OUT}")


if __name__ == "__main__":
    main()
