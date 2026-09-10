#!/usr/bin/env python3
"""Run pretrained stability (ΔΔG) inference on a variant database.

Uses the MegaScale-pretrained stability model (`GAT_mut_processor` with
`weights/MutPred-PPI_stability_pretrain.pt`), NOT the fine-tuned MutPred-PPI, so
that interaction disruption and monomer destabilisation can be compared for the
same variants.

    positive ΔΔG = destabilising    negative ΔΔG = stabilising

Units are kcal/mol, the units of the MegaScale labels the model was fit to; the
model emits ΔΔG directly, so nothing is inverse-transformed.

WHAT CHANGED, AND WHY IT MATTERS FOR THE NUMBERS
------------------------------------------------
This script used to glob `{db}/af3_graphs/*.mat`, call the text before the FIRST
underscore the interactor and the rest the partner, and read chain identity from
`NRR`. It now reads the same canonical inputs as
`run_variant_db_inference.py`:

    rows    `datasets/variant_dbs/{db}_rows.csv.gz`  -- explicit interactor,
            partner and 1-BASED mutation columns, so nothing is split on a
            delimiter that can occur inside an identifier
    graphs  `datasets/variant_dbs/contact_graphs.h5` -- fetched by the two chain
            SEQUENCES and returned already oriented to the requested interactor

Two consequences that are behaviour changes, not refactors:

1. **Self-loops.** The old `.mat` path built `edge_index` from the stored sparse
   matrix and never added the diagonal, while stability PRETRAINING did add it
   (`np.fill_diagonal`). The model was therefore trained with self-attention and
   run without it. `ContactGraphStore.load_edge_index` always includes
   self-loops, so this path now matches training. Every ΔΔG changes slightly.
2. **Chain orientation.** For the pairs whose `.mat` stem contradicted the
   stored chain order, the old path scored the wrong chain. The store orients on
   sequence, so those are now correct.

Both mean the outputs must be REGENERATED rather than appended to; the resume
logic reads the legacy schema so an old file is recognised, but mixing old and
new rows in one TSV would mix two definitions.

Output columns are explicit -- `interactor, partner, mutation, ddg_kcalmol`,
mutation 1-BASED -- matching `run_variant_db_inference.py`. The old composite
`complex_id` is gone for the reason given above.

Usage:
    conda run -n ppi python src/variant_db_inference/run_stability_inference.py \\
        --dataset clinvar --device cuda:0

Output: results/variant_dbs_stability/{dataset}_stability_predictions.tsv
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch

from contact_graphs import ContactGraphStore, check_embedding_lengths  # noqa: E402
from model import GAT_mut_processor  # noqa: E402
from paths import DATA_ROOT, DATASETS_DIR  # noqa: E402
from utils import mutations  # noqa: E402
from variant_db_inference import variant_rows as vr  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
_PUB = _THIS_DIR.parent.parent
_BASE = DATA_ROOT
_MEGASCALE_PRETRAINED = _PUB / "weights" / "MutPred-PPI_stability_pretrain.pt"
_SCALER_PATH = _PUB / "weights" / "mutation_diff_scaler.pkl"
_OUT_DIR = _PUB / "results" / "variant_dbs_stability"

STORE = DATASETS_DIR / "variant_dbs" / "contact_graphs.h5"

DATASET_CONFIGS = {
    db: {
        "rows":        DATASETS_DIR / "variant_dbs" / f"{db}_rows.csv.gz",
        "emb_h5":      _BASE / db / "prott5_embeddings.h5",
        "subgraph_h5": _BASE / db / "prott5_subgraphs.h5",
    }
    for db in ("clinvar", "gnomad", "cosmic", "hgmd", "autism")
}


# The GAT stability predictor is `GAT_mut_processor` in src/model.py -- imported
# above. A verbatim 4th copy of the architecture used to live here; it was
# bit-for-bit identical (state_dict loads strict=True, max output delta 0.0
# over 20 random graphs), so this is a pure de-duplication. The canonical
# class has a 4-arg inference path, so call it exactly as before.


# ── resume ────────────────────────────────────────────────────────────────────

def _load_done(out_path: Path) -> set[tuple[str, str, str]]:
    """Already-scored `(interactor, partner, mutation_1b)` triplets.

    Reads the current explicit-column schema and the legacy
    `complex_id/variant/ddg` one, so an interrupted legacy run is recognised.
    The legacy branch splits on the FIRST underscore, which is safe only because
    UniProt accessions contain none -- it exists to read old files, not to write
    new ones.
    """
    done: set[tuple[str, str, str]] = set()
    if not Path(out_path).exists():
        return done
    with open(out_path) as f:
        header = next(f, None)
        cols = header.rstrip("\n").split("\t") if header else []
        legacy = cols[:1] == ["complex_id"]
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if legacy and len(parts) >= 2:
                inter, _, partner = parts[0].partition("_")
                if partner:
                    done.add((inter, partner, parts[1]))
            elif len(parts) >= 3:
                done.add((parts[0], parts[1], parts[2]))
    return done


def _count_reason(stats: Counter, examples: dict, reason: str, n: int = 1) -> None:
    """Bucket a `check_embedding_lengths` reason, keeping one full example."""
    bucket = reason.split(" has ", 1)[0]
    stats[f"{bucket} length disagrees with its sequence"] += n
    examples.setdefault(bucket, reason)


# ── model ─────────────────────────────────────────────────────────────────────

def predict_stability(node_emb, edge_index_np, model, mut_local_idx,
                      mutation_site_diff_np, device):
    """Return ΔΔG in kcal/mol, or None when the graph cannot be scored.

    No sigmoid: this is a regression head, unlike `inference.model_loader`'s
    `model_predict`, which is why that helper is not reused here.
    """
    try:
        x = torch.tensor(node_emb, dtype=torch.float).to(device)
        ei = torch.tensor(edge_index_np, dtype=torch.long).to(device)
        md = torch.tensor(mutation_site_diff_np, dtype=torch.float).to(device)
        if x.size(0) == 0 or ei.size(1) == 0:
            return None
        if ei.max() >= x.size(0):
            return None
        with torch.no_grad():
            out = model(x, ei, mut_local_idx, md)
        return float(out.squeeze().cpu().numpy())
    except Exception as e:                                     # noqa: BLE001
        print(f"[WARN] predict_stability error: {e}", flush=True)
        return None


def _load_embeddings_h5(h5_path: Path) -> dict[str, np.ndarray]:
    print(f"Loading embeddings from {h5_path} ...", flush=True)
    embs = {}
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            embs[key] = f[key][:]
    print(f"  {len(embs)} embeddings loaded", flush=True)
    return embs


# ── inference paths ───────────────────────────────────────────────────────────

def _run_subgraph(rows, subgraph_h5, model, scaler, device, done, out_file,
                  stats) -> None:
    """Score from the precomputed 2-hop subgraph H5 (compact, low RAM).

    The subgraphs already carry their own `edge_index`, so the store is not
    consulted here. Their H5 group keys are the legacy `{interactor}_{partner}`
    stems and their variant keys are 0-BASED; both are read, never written.
    """
    sg_file = h5py.File(subgraph_h5, "r")
    try:
        for n_rows, r in enumerate(rows, 1):
            interactor, partner, mut_1b = r["interactor"], r["partner"], r["mutation"]
            if (interactor, partner, mut_1b) in done:
                stats["skipped"] += 1
                continue

            cgrp = sg_file.get(f"{interactor}_{partner}")   # legacy H5 layout
            if cgrp is None:
                stats["pair absent from the subgraph H5"] += 1
                continue
            vgrp = cgrp.get(mutations.to_zero_based(mut_1b))   # 0-BASED H5 key
            if vgrp is None:
                stats["variant absent from the subgraph H5"] += 1
                continue
            if "node_emb" not in vgrp or "mut_diff" not in vgrp:
                stats["subgraph entry incomplete"] += 1
                continue

            ddg = predict_stability(
                vgrp["node_emb"][:], vgrp["edge_index"][:], model,
                int(vgrp.attrs["mut_local_idx"]),
                scaler.transform(vgrp["mut_diff"][:].reshape(1, -1)).squeeze(),
                device)
            if ddg is None:
                stats["model returned no score"] += 1
                continue

            out_file.write(f"{interactor}\t{partner}\t{mut_1b}\t{ddg:.4f}\n")
            out_file.flush()
            stats["scored"] += 1

            if n_rows % 50000 == 0:
                print(f"  [{n_rows} rows] scored={stats['scored']} "
                      f"skipped={stats['skipped']}", flush=True)
    finally:
        sg_file.close()


def _run_full_emb(rows, embs, store: ContactGraphStore, model, scaler, device,
                  done, out_file, stats, examples) -> None:
    """Score from bulk-loaded full embeddings, graphs fetched by sequence.

    Rows are grouped by pair so each graph is fetched once, as the old
    file-driven loop did.
    """
    grouped: dict[tuple[str, str], list[str]] = {}
    pair_seqs: dict[tuple[str, str], tuple[str, str]] = {}
    for r in rows:
        k = (r["interactor"], r["partner"])
        grouped.setdefault(k, []).append(r["mutation"])
        pair_seqs.setdefault(k, (r["interactor_sequence"], r["partner_sequence"]))
    print(f"  {sum(len(v) for v in grouped.values()):,} rows over "
          f"{len(grouped):,} pairs", flush=True)

    for pair_idx, (key, muts) in enumerate(sorted(grouped.items()), 1):
        interactor, partner = key
        iseq, pseq = pair_seqs[key]

        if interactor not in embs:
            stats["interactor WT embedding missing"] += len(muts)
            continue
        if partner not in embs:
            stats["partner WT embedding missing"] += len(muts)
            continue
        wt_emb, partner_emb = embs[interactor], embs[partner]

        reason = check_embedding_lengths(
            interactor_seq=iseq, partner_seq=pseq,
            interactor_emb=wt_emb, partner_emb=partner_emb)
        if reason:
            _count_reason(stats, examples, reason, len(muts))
            continue

        # Self-loops included, interactor first -- the store's guarantee, and
        # what stability pretraining assumed. The old path omitted them.
        edge_index_np = store.load_edge_index(interactor=iseq, partner=pseq)
        if edge_index_np is None:
            stats["pair has no graph in the contact-graph store"] += len(muts)
            continue

        for mut_1b in sorted(muts):
            if (interactor, partner, mut_1b) in done:
                stats["skipped"] += 1
                continue

            # ProtT5 H5 keys are 0-BASED; the array index is the same number.
            variant_0b = mutations.to_zero_based(mut_1b)
            mut_idx = mutations.position(variant_0b)
            vt_id = f"{interactor} {variant_0b}"
            if vt_id not in embs:
                stats["variant embedding missing"] += 1
                continue
            vt_emb = embs[vt_id]

            reason = check_embedding_lengths(
                interactor_seq=iseq, partner_seq=pseq, interactor_emb=vt_emb)
            if reason:
                _count_reason(stats, examples, reason)
                continue
            if mut_idx >= wt_emb.shape[0]:
                stats["mutation position past the end of the interactor"] += 1
                continue

            node_emb = np.concatenate([vt_emb, partner_emb], axis=0)
            mut_diff_raw = (vt_emb[mut_idx] - wt_emb[mut_idx]).reshape(1, -1)

            ddg = predict_stability(node_emb, edge_index_np, model, mut_idx,
                                    scaler.transform(mut_diff_raw).squeeze(),
                                    device)
            if ddg is None:
                stats["model returned no score"] += 1
                continue

            out_file.write(f"{interactor}\t{partner}\t{mut_1b}\t{ddg:.4f}\n")
            out_file.flush()
            stats["scored"] += 1

        if pair_idx % 100 == 0:
            print(f"  [{pair_idx}/{len(grouped)}] scored={stats['scored']} "
                  f"skipped={stats['skipped']}", flush=True)


def run_dataset(dataset: str, device: torch.device, model, scaler,
                out_path: Path, store_path: Path) -> None:
    cfg = DATASET_CONFIGS[dataset]
    rows_path = Path(cfg["rows"])
    if not rows_path.exists():
        print(f"[SKIP] {dataset}: row table not found at {rows_path} -- run "
              f"build_variant_db_tables.py", flush=True)
        return

    # Resolve the embedding source BEFORE touching the output, so a skipped
    # dataset does not leave a header-only TSV behind for the next run to
    # mistake for a completed one.
    subgraph_h5 = Path(cfg["subgraph_h5"])
    emb_h5 = Path(cfg["emb_h5"])
    if not subgraph_h5.exists() and not emb_h5.exists():
        print(f"[SKIP] {dataset}: neither {subgraph_h5} nor {emb_h5} exists",
              flush=True)
        return

    stats: Counter = Counter()
    examples: dict[str, str] = {}
    done = _load_done(out_path)
    print(f"{dataset}: {len(done)} already scored", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = vr.iter_table_rows(dataset, rows_path, stats=stats)

    with open(out_path, "a") as out_file:
        if len(done) == 0:
            out_file.write("interactor\tpartner\tmutation\tddg_kcalmol\n")

        if subgraph_h5.exists():
            print(f"  subgraph H5 found: {subgraph_h5}", flush=True)
            _run_subgraph(rows, subgraph_h5, model, scaler, device, done,
                          out_file, stats)
        else:
            print(f"  using full embedding H5: {emb_h5}", flush=True)
            embs = _load_embeddings_h5(emb_h5)
            with ContactGraphStore(store_path) as store:
                _run_full_emb(rows, embs, store, model, scaler, device, done,
                              out_file, stats, examples)

    print(f"  Done: {stats['scored']} new, {stats['skipped']} skipped", flush=True)
    print("  not scored, by reason:", flush=True)
    for k, v in stats.most_common():
        if k not in ("scored", "skipped"):
            print(f"    {k}: {v:,}", flush=True)
    for ex in examples.values():
        print(f"    e.g. {ex}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=list(DATASET_CONFIGS.keys()) + ["all"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--store", default=str(STORE),
                    help="Contact-graph store (default: "
                         "datasets/variant_dbs/contact_graphs.h5)")
    ap.add_argument("--out-dir", default=str(_OUT_DIR))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    print(f"Device: {device}", flush=True)

    print(f"Loading pretrained stability model from {_MEGASCALE_PRETRAINED}", flush=True)
    model = GAT_mut_processor(input_dim=1024).to(device)
    state = torch.load(_MEGASCALE_PRETRAINED, weights_only=True, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("  Model loaded", flush=True)

    print(f"Loading scaler from {_SCALER_PATH}", flush=True)
    scaler = joblib.load(_SCALER_PATH)
    print("  Scaler loaded", flush=True)

    datasets = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        out_path = Path(args.out_dir) / f"{ds}_stability_predictions.tsv"
        print(f"\n=== {ds} → {out_path}", flush=True)
        run_dataset(ds, device, model, scaler, out_path, Path(args.store))


if __name__ == "__main__":
    main()
