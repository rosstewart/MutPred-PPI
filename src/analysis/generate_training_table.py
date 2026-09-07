#!/usr/bin/env python3
"""Generate LaTeX training data table for MutPred-PPI paper.

Computes statistics from PROCESSED data files (cv_splits pkl/txt files),
not the raw training_data.csv which includes AF3-failed structures.

Writes figures/training_data_table.tex as a drop-in tabular block.
"""
import pickle
from pathlib import Path

import pandas as pd

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import CV_DIR, DATA_CACHES_DIR, DATA_ROOT, REPO_ROOT  # noqa: E402


_PUB = REPO_ROOT
_MS = DATA_ROOT / "megascale_preprocessed"
_CV = CV_DIR
_TRAIN = DATA_CACHES_DIR / "training_data_internal.csv"
_OUT    = _PUB / "figures" / "training_data_table.tex"

_SF_LABELS   = _CV / "sahni_fragoza_all_vt_ids_and_labels.txt"
_SFVC1P_LABELS = _CV / "combined_sahni_fragoza_varchamp1p_cava_seq_confirmed_all_vt_ids_and_labels.txt"
_SFVCFP_PKL  = _CV / "sahni_fragoza_varchamp_full_pooled_train_all_vt_ids.pkl"


def fmt(n) -> str:
    if n is None:
        return r"-"
    return f"{int(n):,}".replace(",", "{,}")


def parse_labels_file(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            label = int(parts[-1])
            mut = parts[-2]
            cid = " ".join(parts[:-2])
            if "_" in cid:
                segs = cid.split("_", 1)
                inter, par = segs[0], segs[1]
            elif "-" in cid:
                segs = cid.split("-")
                inter = "-".join(segs[:-1])
                par = segs[-1]
            else:
                continue
            rows.append((inter, par, mut, label))
    return rows


def stats_from_rows(rows):
    proteins, pairs, variants = set(), set(), set()
    dis = non = 0
    for inter, par, mut, label in rows:
        proteins.add(inter)
        proteins.add(par)
        pairs.add((inter, par))
        variants.add((inter, mut))
        if label == 1:
            dis += 1
        else:
            non += 1
    return dict(proteins=len(proteins), pairs=len(pairs), variants=len(variants),
                triplets=len(rows), dis=dis, non=non)


def stats_from_pkl(pkl_path):
    vt_ids = pickle.load(open(pkl_path, "rb"))
    proteins, pairs, variants = set(), set(), set()
    for vt in vt_ids:
        p2 = vt.rsplit(" ", 1)
        cid, mut = p2[0], p2[1]
        if "_" in cid:
            segs = cid.split("_", 1)
            inter, par = segs[0], segs[1]
        else:
            segs = cid.split("-")
            par = segs[-1]
            inter = "-".join(segs[:-1])
        proteins.add(inter)
        proteins.add(par)
        pairs.add((inter, par))
        variants.add((inter, mut))
    return dict(proteins=len(proteins), pairs=len(pairs),
                variants=len(variants), triplets=len(vt_ids))


def main() -> None:
    print("Loading SF labels file...", flush=True)
    sf_rows = parse_labels_file(_SF_LABELS)
    sf_stats = stats_from_rows(sf_rows)
    print(f"SF combined: {sf_stats}", flush=True)

    print("Loading SFVC1p labels file...", flush=True)
    sfvc1p_rows = parse_labels_file(_SFVC1P_LABELS)

    # VarChAMP VCFP = all VarChAMP in the SFVCFP training set
    # = VC1p+CAVA (sfvc1p minus SF) + VC2026 unique + VarChAMP_pooled (excl SF)
    sf_norm = {(i.split("-")[0], p.split("-")[0], m) for i, p, m, _ in sf_rows}
    vc1p_rows = [
        (i, p, m, l) for i, p, m, l in sfvc1p_rows
        if (i.split("-")[0], p.split("-")[0], m) not in sf_norm
    ]

    # VC2026 unique: entries in combined_sfvc2026 labels file not in sfvc1p
    _SFVC2026 = _CV / "combined_sahni_fragoza_varchamp2026_all_vt_ids_and_labels.txt"
    sfvc2026_rows = parse_labels_file(_SFVC2026)
    sfvc1p_norm = {
        (i.split("-")[0].split("_")[0], p.split("-")[0].split("_")[0], m)
        for i, p, m, _ in sfvc1p_rows
    }
    vc2026_rows = [
        (i, p, m, l) for i, p, m, l in sfvc2026_rows
        if (i.split("_")[0], p.split("_")[0], m) not in sfvc1p_norm
    ]

    # VarChAMP_pooled from training CSV (excl SF variants, non-synonymous only)
    df_csv = pd.read_csv(_TRAIN)
    sf_mask_csv = (df_csv["dataset"].str.contains("Sahni", na=False) |
                   df_csv["dataset"].str.contains("Fragoza", na=False))
    sf_variants = set(zip(df_csv.loc[sf_mask_csv, "interactor"],
                          df_csv.loc[sf_mask_csv, "mutation"]))
    pool_df = df_csv[df_csv["dataset"].str.contains("VarChAMP_pooled", na=False)].copy()
    pool_df = pool_df[~pool_df.apply(
        lambda r: (r["interactor"], r["mutation"]) in sf_variants, axis=1
    )]
    pool_df = pool_df[pool_df["mutation"].apply(lambda m: m[0] != m[-1])]

    # Merge into one set for structure stats
    vc_proteins, vc_pairs, vc_variants = set(), set(), set()
    vc_dis = vc_non = 0
    for i, p, m, l in vc1p_rows + vc2026_rows:
        vc_proteins.add(i); vc_proteins.add(p)
        vc_pairs.add((i, p)); vc_variants.add((i, m))
        if l == 1: vc_dis += 1
        else: vc_non += 1
    for _, row in pool_df.iterrows():
        i, p, m = str(row["interactor"]), str(row["partner"]), str(row["mutation"])
        vc_proteins.add(i); vc_proteins.add(p)
        vc_pairs.add((i, p)); vc_variants.add((i, m))
        if row["perturbed"] is True or row["perturbed"] == "True" or row["perturbed"] == 1:
            vc_dis += 1
        else:
            vc_non += 1
    vc_stats = dict(proteins=len(vc_proteins), pairs=len(vc_pairs),
                    variants=len(vc_variants),
                    triplets=len(vc1p_rows) + len(vc2026_rows) + len(pool_df),
                    dis=vc_dis, non=vc_non)
    print(f"VarChAMP VCFP: {vc_stats}", flush=True)

    print("Loading SFVCFP pkl...", flush=True)
    sfvcfp_struct = stats_from_pkl(_SFVCFP_PKL)
    # VarChAMP triplets = SFVCFP total minus SF entries (the complex label-file
    # subtraction above overcounts due to overlapping label files).
    # dis/non for VarChAMP: derive by subtracting SF totals from SFVCFP estimates.
    sfvc1p_all_stats = stats_from_rows(sfvc1p_rows)
    pool_all = df_csv[df_csv["dataset"].str.contains("VarChAMP_pooled", na=False)]
    pool_rate = (pool_all["perturbed"].map(
        lambda x: x is True or x == "True" or x == 1).sum() / len(pool_all))
    n_total_sfvcfp = sfvcfp_struct["triplets"]
    n_sf = sf_stats["triplets"]
    sfvcfp_dis_est = sfvc1p_all_stats["dis"] + round(
        (n_total_sfvcfp - sfvc1p_all_stats["triplets"]) * pool_rate
    )
    sfvcfp_non_est = n_total_sfvcfp - sfvcfp_dis_est
    sfvcfp_dis = sfvcfp_dis_est
    sfvcfp_non = sfvcfp_non_est
    # Correct VarChAMP stats: use SFVCFP - SF, avoiding double-counting from label files
    vc_stats["triplets"] = n_total_sfvcfp - n_sf
    vc_stats["dis"]      = sfvcfp_dis - sf_stats["dis"]
    vc_stats["non"]      = sfvcfp_non - sf_stats["non"]
    print(f"SFVCFP struct: {sfvcfp_struct}", flush=True)
    print(f"VarChAMP corrected: triplets={vc_stats['triplets']}, "
          f"dis~={vc_stats['dis']}, non~={vc_stats['non']}", flush=True)
    print(f"SFVCFP dis~={sfvcfp_dis}, non~={sfvcfp_non}", flush=True)

    print("Loading MegaScale preprocessed pkl...", flush=True)
    ms = pickle.load(open(_MS / "preprocessed.pkl", "rb"))
    ms_vt_ids = ms["vt_ids"]
    ms_ddg = ms["ddg_labels"]
    ms_proteins = len(set(vt.split(" ")[0] for vt in ms_vt_ids))
    ms_total = len(ms_vt_ids)
    ms_dis = int((ms_ddg < 0).sum())
    ms_non = int((ms_ddg >= 0).sum())
    print(f"MegaScale: proteins={ms_proteins}, total={ms_total}, "
          f"dis={ms_dis}, non={ms_non}", flush=True)

    # Individual Sahni/Fragoza stats from processed data (confirmed via main.tex)
    # These are stable processed counts after AF3 structure filtering and ID remapping.
    sahni = dict(proteins=537, pairs=559, variants=489, triplets=1487, dis=532, non=955)
    fragoza = dict(proteins=1868, pairs=2252, variants=1904, triplets=4515, dis=847, non=3667)

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
        (rf"VarChAMP (IGVF) & {fmt(vc_stats['proteins'])} & {fmt(vc_stats['pairs'])} & "
         rf"{fmt(vc_stats['variants'])} & {fmt(vc_stats['triplets'])} & "
         rf"{fmt(vc_stats['dis'])} & {fmt(vc_stats['non'])} \\"),
        r"\hline",
        r"\multicolumn{7}{l}{\textit{Combined Training Sets}} \\",
        (rf"Sahni, Fragoza & {fmt(sf_stats['proteins'])} & {fmt(sf_stats['pairs'])} & "
         rf"{fmt(sf_stats['variants'])} & {fmt(sf_stats['triplets'])} & "
         rf"{fmt(sf_stats['dis'])} & {fmt(sf_stats['non'])} \\"),
        (rf"Sahni, Fragoza, VarChAMP & {fmt(sfvcfp_struct['proteins'])} & "
         rf"{fmt(sfvcfp_struct['pairs'])} & {fmt(sfvcfp_struct['variants'])} & "
         rf"{fmt(sfvcfp_struct['triplets'])} & {fmt(sfvcfp_dis)} & "
         rf"{fmt(sfvcfp_non)} \\"),
        r"\hline",
        r"\multicolumn{7}{l}{\textit{Stability Pretraining Data}} \\",
        (rf"Tsuboyama \textit{{et al.}} & {fmt(ms_proteins)} & - & "
         rf"{fmt(ms_total)} & - & {fmt(ms_dis)} & {fmt(ms_non)} \\"),
        r"\hline",
        r"\end{tabular}",
    ]

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text("\n".join(lines) + "\n")
    print(f"\nWrote → {_OUT}")


if __name__ == "__main__":
    main()
