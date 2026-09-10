#!/usr/bin/env python
"""Run MutPred-PPI inference on a variant database using precomputed ProtT5 embeddings.

Rows come from `datasets/variant_dbs/{db}_rows.csv.gz` — one
`(interactor, partner, mutation)` triplet per prediction — and contact graphs
from the content-addressed store, fetched by the two chain SEQUENCES. The
previous version globbed `{db}/af3_graphs/*.mat`, called the first token of the
filename the interactor and sliced at the stored `NRR`; for 121 clinvar / 121
cosmic / 116 gnomad / 29 hgmd files the stored chain order contradicts the
filename, so those complexes were scored on the wrong chain. The store returns
the graph already oriented to the requested interactor, so `NRR` is gone.

Prerequisites:
1. ProtT5 embeddings precomputed with precompute_prott5.py (and, for the large
   databases, compressed with compress_to_subgraphs.py)
2. `datasets/variant_dbs/contact_graphs.h5` and `{db}_rows.csv.gz`
   (build_variant_db_tables.py)
3. Trained model checkpoints in weights/ (the SFVCFP model — variant-DB inference
   is not a blind test, so the model trained on the most data is used)

Output TSV carries EXPLICIT columns: `interactor`, `partner`, `mutation` (1-BASED),
`score`. The old composite `complex_id` = `{interactor}_{partner}` is gone --
downstream code had to split it back apart on `_`, and every such split is a
latent bug when an identifier contains the delimiter. Resume keys are tuples for
the same reason.

Usage (nohup recommended for large datasets):
    nohup conda run -n ppi python run_variant_db_inference.py \\
        --dataset gnomad --device cuda:0 \\
        --embeddings-h5 $MUTPRED_DATA_ROOT/gnomad/prott5_embeddings.h5 \\
        >> inference_gnomad.log 2>&1 &

NOTE: HGMD and COSMIC datasets require licensed input data that cannot be
redistributed.  The scripts that generate their inputs take the licensed source
files as input.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch

# Resolve the models directory relative to this file
_THIS_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _THIS_DIR.parent.parent / "weights"
_SCALER_PATH = _MODELS_DIR / "mutation_diff_scaler.pkl"

from contact_graphs import ContactGraphStore, check_embedding_lengths  # noqa: E402
from inference.utils.model_loader import get_models, model_predict, model_predict_subgraph  # noqa: E402
from paths import DATA_ROOT, DATASETS_DIR  # noqa: E402
from variant_db_inference import variant_rows as vr  # noqa: E402
from utils import mutations  # noqa: E402


# ── dataset path registry ─────────────────────────────────────────────────────

_BASE = DATA_ROOT
STORE = DATASETS_DIR / "variant_dbs" / "contact_graphs.h5"
DATASET_CONFIGS = {
    db: {
        "default_emb_h5":      _BASE / db / "prott5_embeddings.h5",
        "default_subgraph_h5": _BASE / db / "prott5_subgraphs.h5",
        "default_out":         _BASE / db / "mutpred_ppi_predictions.tsv",
    }
    for db in ("clinvar", "gnomad", "hgmd", "cosmic", "autism", "neurodev")
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_embeddings_h5(h5_path: str) -> dict[str, np.ndarray]:
    print(f"Loading embeddings from {h5_path} ...", flush=True)
    embs: dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            embs[key] = f[key][:]
    print(f"  {len(embs)} sequences loaded", flush=True)
    return embs


def _load_done(out_path: str) -> set[tuple[str, str, str]]:
    """Already-scored `(interactor, partner, mutation)` triplets, for resume.

    Reads both the current explicit-column schema and the legacy
    `complex_id/variant/score` one, so an interrupted legacy run can still be
    resumed. The legacy branch splits on the FIRST underscore, which is safe only
    because UniProt accessions contain none.
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
    """Bucket a `check_embedding_lengths` reason, keeping one full example.

    The reason carries the offending lengths, which would give the Counter a key
    per complex; the text before them is the class of failure.
    """
    bucket = reason.split(" has ", 1)[0]
    stats[f"{bucket} length disagrees with its sequence"] += n
    examples.setdefault(bucket, reason)


def _report(stats: Counter, examples: dict, out_path: str) -> None:
    print(f"\nDone. {stats['scored']} new predictions written to {out_path}", flush=True)
    print(f"  skipped (already done): {stats['skipped']}", flush=True)
    print("  not scored, by reason:", flush=True)
    for k, v in stats.most_common():
        if k not in ("scored", "skipped", "rows read"):
            print(f"    {k}: {v:,}", flush=True)
    for ex in examples.values():
        print(f"    e.g. {ex}", flush=True)


# ── inference ─────────────────────────────────────────────────────────────────

def run_inference(
    db: str,
    rows_path: str,
    store_path: str,
    emb_h5: str,
    out_path: str,
    models_dir: str,
    device_str: str,
    subgraph_h5: str | None = None,
) -> None:
    device = torch.device(device_str if device_str else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}", flush=True)

    models = get_models(models_dir, device)
    print(f"Loaded {len(models)} MutPred-PPI models from {models_dir}", flush=True)

    scaler_path = Path(models_dir) / "mutation_diff_scaler.pkl"
    scaler = joblib.load(str(scaler_path))
    print(f"Loaded scaler from {scaler_path}", flush=True)

    # Prefer subgraph H5 if available — avoids loading multi-TB full embeddings into RAM
    use_subgraph = False
    if subgraph_h5 and Path(subgraph_h5).exists():
        use_subgraph = True
        print(f"Subgraph H5 found: {subgraph_h5} — using compact 2-hop inference mode",
              flush=True)
    elif not Path(emb_h5).exists():
        print("[ERROR] Neither subgraph H5 nor embedding H5 found", flush=True)
        sys.exit(1)
    else:
        print(f"Using full embedding H5: {emb_h5}", flush=True)

    done = _load_done(out_path)
    print(f"  {len(done)} already scored — skipping", flush=True)

    stats: Counter = Counter()
    examples: dict[str, str] = {}
    print(f"Rows: {rows_path}", flush=True)
    rows = vr.iter_table_rows(db, rows_path, stats=stats)

    out_file = open(out_path, "a")
    if len(done) == 0:
        out_file.write("interactor\tpartner\tmutation\tscore\n")

    if use_subgraph:
        # The 2-hop subgraphs already carry their own graph, so the store is only
        # needed on the full-embedding path.
        _run_inference_subgraph(rows, subgraph_h5, scaler, models, device,
                                done, out_file, stats, examples)
    else:
        # Legacy path: bulk-load full embeddings (only feasible for small datasets)
        embs = _load_embeddings_h5(emb_h5)
        store = ContactGraphStore(store_path)
        _run_inference_full_emb(rows, embs, store, scaler, models, device,
                                done, out_file, stats, examples)
        store.close()

    out_file.close()
    _report(stats, examples, out_path)


def _run_inference_subgraph(rows, subgraph_h5, scaler, models, device,
                            done, out_file, stats, examples):
    """Inference using pre-computed 2-hop subgraph H5 (compact, low RAM)."""
    sg_file = h5py.File(subgraph_h5, "r")

    for n_rows, r in enumerate(rows, 1):
        interactor, partner = r["interactor"], r["partner"]
        mut_1b = r["mutation"]
        if (interactor, partner, mut_1b) in done:
            stats["skipped"] += 1
            continue

        variant = vr.to_zero_based(mut_1b)   # subgraph H5 variant keys are 0-based
        cgrp = sg_file.get(f"{interactor}_{partner}")   # H5 group key, legacy layout
        if cgrp is None:
            stats["pair absent from the subgraph H5"] += 1
            continue
        vgrp = cgrp.get(variant)
        if vgrp is None:
            stats["variant absent from the subgraph H5"] += 1
            continue
        if "node_emb" not in vgrp or "mut_diff" not in vgrp:
            stats["subgraph entry incomplete"] += 1
            continue

        node_emb      = vgrp["node_emb"][:]
        edge_index_np = vgrp["edge_index"][:]
        mut_diff_raw  = vgrp["mut_diff"][:].reshape(1, -1)
        mut_local_idx = int(vgrp.attrs["mut_local_idx"])

        mutation_site_diff = scaler.transform(mut_diff_raw).squeeze()

        score = model_predict_subgraph(
            node_emb, edge_index_np, models, mut_local_idx,
            mutation_site_diff, device,
        )
        if score is None:
            stats["model returned no score"] += 1
            continue

        out_file.write(f"{interactor}\t{partner}\t{mut_1b}\t{float(score):.6f}\n")
        out_file.flush()
        stats["scored"] += 1

        if n_rows % 50000 == 0:
            print(f"[{n_rows} rows] scored={stats['scored']}  "
                  f"skipped={stats['skipped']}", flush=True)

    sg_file.close()


def _run_inference_full_emb(rows, embs, store, scaler, models, device,
                            done, out_file, stats, examples):
    """Legacy inference path using bulk-loaded full embeddings.

    Rows are grouped by pair so each contact graph is fetched once, as the
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

        edge_mat = store.load_dense(interactor=iseq, partner=pseq)
        if edge_mat is None:
            stats["pair has no graph in the contact-graph store"] += len(muts)
            continue

        for mut_1b in sorted(muts):
            if (interactor, partner, mut_1b) in done:
                stats["skipped"] += 1
                continue

            variant = vr.to_zero_based(mut_1b)   # ProtT5 keys are 0-based
            mut_idx = mutations.position(variant)   # H5 keys are 0-based
            vt_id = f"{interactor} {variant}"
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

            combined_emb = np.concatenate([vt_emb, partner_emb], axis=0)
            mut_diff_raw = (vt_emb[mut_idx] - wt_emb[mut_idx]).reshape(1, -1)
            mutation_site_diff = scaler.transform(mut_diff_raw).squeeze()

            score = model_predict(
                combined_emb, edge_mat, models, mut_idx, mutation_site_diff, device
            )
            if score is None:
                stats["model returned no score"] += 1
                continue

            out_file.write(f"{interactor}\t{partner}\t{mut_1b}\t{float(score):.6f}\n")
            out_file.flush()
            stats["scored"] += 1

        if pair_idx % 100 == 0:
            print(f"[{pair_idx}/{len(grouped)}] scored={stats['scored']}  "
                  f"skipped={stats['skipped']}", flush=True)


def assert_all_data_model(models_dir: str) -> None:
    """Refuse to score a variant repository with anything but the all-data model.

    `get_models()` falls back to a fold-ensemble glob
    (`MutPred-PPI_*_megascale_all_*.pt`) when `MutPred-PPI.pt` is absent from
    `models_dir` -- a fallback meant for GCV reproducibility, not variant-DB
    scoring. Pointing `--models-dir` at `weights/folds/` (the Sahni+Fragoza-only
    per-fold checkpoints) would silently satisfy that glob and score every
    variant repository with the wrong model -- exactly how the now-archived
    `results/variant_dbs/` tree (see archive/results_stale/variant_dbs) came to
    exist alongside the correct `results/variant_dbs_all_data/`. There must be
    only one variant-DB results tree, scored by one model.
    """
    primary = Path(models_dir) / "MutPred-PPI.pt"
    if not primary.is_file():
        raise FileNotFoundError(
            f"{primary} not found. Variant-database inference must use the single "
            f"all-data model (trained on sahni_fragoza_varchamp_all_mapped090826), "
            f"never a fold ensemble or any other checkpoint set -- point --models-dir "
            f"at the directory containing MutPred-PPI.pt (default: weights/)."
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    if args.graph_dir:
        print("[note] --graph-dir is ignored: graphs now come from --store, "
              "addressed by sequence", flush=True)

    # --dataset is now required: rows and WT sequences are per-database.
    if not args.dataset:
        print("ERROR: --dataset is required (explicit path args still override "
              "the defaults it sets).", file=sys.stderr)
        sys.exit(1)

    cfg = DATASET_CONFIGS[args.dataset]
    emb_h5      = str(args.embeddings_h5 or cfg["default_emb_h5"])
    subgraph_h5 = str(args.subgraphs_h5  or cfg["default_subgraph_h5"])
    out_path    = str(args.out           or cfg["default_out"])
    rows_path   = str(args.rows          or vr.table_path(args.dataset))
    if not Path(rows_path).exists():
        print(f"ERROR: {rows_path} not found — run build_variant_db_tables.py "
              f"--db {args.dataset}", file=sys.stderr)
        sys.exit(1)

    models_dir = str(args.models_dir or _MODELS_DIR)
    assert_all_data_model(models_dir)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    run_inference(
        db=args.dataset,
        rows_path=rows_path,
        store_path=str(args.store),
        emb_h5=emb_h5,
        out_path=out_path,
        models_dir=models_dir,
        device_str=args.device,
        subgraph_h5=subgraph_h5,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run MutPred-PPI inference on variant databases")
    p.add_argument("--dataset", choices=list(DATASET_CONFIGS),
                   help="Named dataset (sets default paths; overridden by explicit path args)")
    p.add_argument("--rows",
                   help="Row table (default: datasets/variant_dbs/{dataset}_rows.csv.gz)")
    p.add_argument("--store", default=str(STORE),
                   help=f"Contact-graph HDF5 store (default: {STORE})")
    p.add_argument("--embeddings-h5",
                   help="Path to full per-protein ProtT5 H5 (legacy; not needed when subgraphs H5 exists)")
    p.add_argument("--subgraphs-h5",
                   help="Path to compact 2-hop subgraph H5 (preferred; auto-detected per dataset)")
    p.add_argument("--out",
                   help="Output TSV path (default: dataset-specific path under /data)")
    p.add_argument("--models-dir", default=None,
                   help=f"Directory containing .pt model files and scaler (default: {_MODELS_DIR})")
    p.add_argument("--device", default="",
                   help="PyTorch device string (e.g. 'cuda:0'). Defaults to auto-detect.")
    p.add_argument("--graph-dir", help=argparse.SUPPRESS)  # accepted, ignored
    main(p.parse_args())
