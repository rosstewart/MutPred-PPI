#!/usr/bin/env python
"""DDMut-PPI inference script for all four interaction-loss datasets.

Calls the DDMut-PPI REST API (https://biosig.lab.uq.edu.au/ddmut_ppi/api/)
to get ΔΔG predictions, with per-complex batching (≤500 mutations per job),
resume support, and exponential-backoff polling.

Results are saved as a flat numpy array in fold iteration order — matching
the format expected by the eval notebook.

Workflow:
  1. Load base fold_splits and labels for the dataset.
  2. Group all unique test variants by PDB complex.
  3. Submit /api/list jobs (one job per complex, ≤500 mutations each).
  4. Poll all jobs until DONE; parse predictions into a cache dict.
  5. Build flat pred array in fold-test order; save as {dataset}_DDMutPPI_preds.npy.

Resume: if {outdir}/DDMutPPI_{dataset}_cache.pkl exists, already-computed
variants are skipped and their job_ids (in {outdir}/DDMutPPI_{dataset}_jobs.json)
are re-polled if still pending.

Usage:
    conda run -n ppi python ddmutppi_preds.py \\
        --dataset sahni_fragoza --outdir ./results/
    conda run -n ppi python ddmutppi_preds.py \\
        --dataset sahni_fragoza_varchamp1p_cava \\
        --outdir ./results/ --poll-interval 60
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import CV_DIR, DATA_ROOT  # noqa: E402


# ── fixed paths ───────────────────────────────────────────────────────────────
_CV_DIR = CV_DIR
_AF3_PDBS = DATA_ROOT / "three_datasets_af3_models" / "pdbs"
_SAHNI_PDBS = DATA_ROOT / "sahni_pdbs"
_ALL_TO_UNIPROT = DATA_ROOT / "all_to_uniprot.pkl"
_API_BASE = "https://biosig.lab.uq.edu.au/ddmut_ppi/api"


# ── dataset configuration ─────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    name: str
    fold_splits_file: str
    labels_file: str
    use_sahni_pdbs: bool


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "sahni": DatasetConfig(
        name="sahni",
        fold_splits_file="fold_splits.pkl",
        labels_file="all_vt_ids_and_labels.txt",
        use_sahni_pdbs=True,
    ),
    "sahni_fragoza": DatasetConfig(
        name="sahni_fragoza",
        fold_splits_file="sahni_fragoza_train_fold_splits.pkl",
        labels_file="sahni_fragoza_all_vt_ids_and_labels.txt",
        use_sahni_pdbs=False,
    ),
    "sahni_varchamp1p_cava": DatasetConfig(
        name="sahni_varchamp1p_cava",
        fold_splits_file="sahni_varchamp1p_cava_train_fold_splits.pkl",
        labels_file="combined_sahni_varchamp1p_cava_seq_confirmed_all_vt_ids_and_labels.txt",
        use_sahni_pdbs=False,
    ),
    "sahni_fragoza_varchamp1p_cava": DatasetConfig(
        name="sahni_fragoza_varchamp1p_cava",
        fold_splits_file="sahni_fragoza_varchamp1p_cava_train_fold_splits.pkl",
        labels_file="combined_sahni_fragoza_varchamp1p_cava_seq_confirmed_all_vt_ids_and_labels.txt",
        use_sahni_pdbs=False,
    ),
}


# ── ID helpers ────────────────────────────────────────────────────────────────

def get_gene_name(s: str) -> str:
    if s.startswith(("NP_", "np_")):
        return s
    if "_" not in s:
        return s
    return "_".join(s.split("_")[:-1])


def split_wt_id(wt_id: str) -> tuple[str, str]:
    if wt_id.startswith(("NP_", "np_")):
        return "_".join(wt_id.split("_")[:2]), "_".join(wt_id.split("_")[2:])
    delim = "_" if "_" in wt_id else "-"
    parts = wt_id.split(delim)
    if len(parts) == 2:
        return (parts[0], parts[1])
    for idx, part in enumerate(parts):
        try:
            int(part)
            split_at = idx + 1
            if split_at == len(parts):
                split_at = 1
            return (delim.join(parts[:split_at]), delim.join(parts[split_at:]))
        except ValueError:
            continue
    raise ValueError(f"Could not split wt_id: {wt_id}")


def _norm(s: str) -> str:
    return s.lower().replace("-", "_")


def _build_u2g(all_to_uniprot: dict) -> dict[str, list[str]]:
    inv: dict[str, list[str]] = {}
    for gene, uniprot in all_to_uniprot.items():
        inv.setdefault(uniprot, []).append(gene)
    return inv


# ── PDB resolution ────────────────────────────────────────────────────────────

def _try_af3(g1: str, g2: str) -> Path:
    return _AF3_PDBS / f"fold_{_norm(g1)}_{_norm(g2)}_model_0.pdb"


def resolve_pdb(complex_id: str, cfg: DatasetConfig,
                u2g: dict[str, list[str]]) -> Path:
    part_1, part_2 = split_wt_id(complex_id)
    if cfg.use_sahni_pdbs:
        return _SAHNI_PDBS / f"fold_{_norm(part_1)}_{_norm(part_2)}_model_0.pdb"
    g1, g2 = get_gene_name(part_1), get_gene_name(part_2)
    cand = _try_af3(g1, g2)
    if cand.exists():
        return cand
    b1, b2 = part_1.split("-")[0], part_2.split("-")[0]
    for ga in u2g.get(b1, [g1]):
        for gb in u2g.get(b2, [g2]):
            c2 = _try_af3(get_gene_name(ga), get_gene_name(gb))
            if c2.exists():
                return c2
    return cand


# ── DDMut-PPI API calls ───────────────────────────────────────────────────────

def _submit_list_job(pdb_path: Path, mutations: list[str],
                     session: requests.Session) -> str | None:
    """POST /api/list with a PDB file and mutation list. Returns job_id or None."""
    mutation_text = "\n".join(mutations)
    try:
        resp = session.post(
            f"{_API_BASE}/list",
            files={
                "pdb_file": ("structure.pdb", open(pdb_path, "rb"), "chemical/x-pdb"),
                "mutations_list": ("mutations.txt", mutation_text.encode(), "text/plain"),
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("job_id")
    except Exception as exc:
        print(f"  SUBMIT ERROR for {pdb_path.name}: {exc}", flush=True)
        return None


def _poll_job(job_id: str, session: requests.Session) -> dict | None:
    """GET /api/list with job_id as form body. Returns parsed result dict when DONE, else None."""
    try:
        resp = session.get(
            f"{_API_BASE}/list",
            data={"job_id": job_id},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("status") == "DONE":
            return data
        if isinstance(data, dict) and data.get("message") == "RUNNING":
            return None
        # unexpected response — treat as done if it has prediction keys
        if isinstance(data, dict) and any(k not in ("job_id", "status", "results_page")
                                           for k in data):
            return data
        return None
    except Exception as exc:
        print(f"  POLL ERROR job {job_id}: {exc}", flush=True)
        return None


def _variant_to_api_mut(variant: str) -> str:
    """Convert 0-indexed variant (e.g. 'S99P') to API mutation string 'A S100P' (1-based)."""
    wt, pos, mt = variant[0], int(variant[1:-1]), variant[-1]
    return f"A {wt}{pos + 1}{mt}"


def _api_key_to_variant(api_key: str) -> str:
    """Convert API result key (e.g. 'A_S100P') back to 0-indexed variant 'S99P'."""
    mut = api_key.split("_", 1)[-1] if "_" in api_key else api_key
    wt, pos_str, mt = mut[0], mut[1:-1], mut[-1]
    return f"{wt}{int(pos_str) - 1}{mt}"


def _parse_result(result: dict) -> dict[str, dict]:
    """Extract {variant (0-indexed) → {"pred": ddG, "outcome": binary_int}} from /api/list result.

    "outcome" is 1 if the mutation is predicted destabilizing/disruptive
    (outcome field contains "Disruptive", "Destabilizing", or "Increasing"),
    0 otherwise. -1 if no outcome field is present.
    """
    _DISRUPTIVE_KEYWORDS = {"disruptive", "destabilizing", "increasing"}
    parsed: dict[str, dict] = {}
    for key, val in result.items():
        if key in ("job_id", "status", "results_page"):
            continue
        if isinstance(val, dict) and "prediction" in val:
            mut_key = _api_key_to_variant(key)
            outcome_str = str(val.get("outcome", "")).lower()
            binary = (1 if any(kw in outcome_str for kw in _DISRUPTIVE_KEYWORDS)
                      else (0 if outcome_str else -1))
            parsed[mut_key] = {"pred": float(val["prediction"]), "outcome": binary}
    return parsed


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    cfg    = DATASET_CONFIGS[args.dataset]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cache_path = outdir / f"DDMutPPI_{cfg.name}_cache.pkl"
    jobs_path  = outdir / f"DDMutPPI_{cfg.name}_jobs.json"
    out_npy    = outdir / f"{cfg.name}_DDMutPPI_preds.npy"
    binary_check = outdir / f"{cfg.name}_DDMutPPI_binary_labels.npy"

    if out_npy.exists() and binary_check.exists() and not args.overwrite:
        print(f"Output already exists: {out_npy}  (use --overwrite to rerun)", flush=True)
        return

    # load or initialise cache (dict: vt_id → {"pred": float, "outcome": int})
    cache: dict[str, dict] = {}
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            raw = pickle.load(f)
        # backward-compatible: old cache may store plain floats
        for k, v in raw.items():
            cache[k] = v if isinstance(v, dict) else {"pred": v, "outcome": -1}
        print(f"Loaded cache: {len(cache)} variants already computed", flush=True)

    # load or initialise pending jobs (list of {job_id, complex_id, variants})
    pending_jobs: list[dict] = []
    if jobs_path.exists():
        with open(jobs_path) as f:
            pending_jobs = json.load(f)
        print(f"Loaded {len(pending_jobs)} pending jobs from {jobs_path}", flush=True)

    with open(_ALL_TO_UNIPROT, "rb") as f:
        all_to_uniprot = pickle.load(f)
    u2g = _build_u2g(all_to_uniprot)

    with open(_CV_DIR / cfg.fold_splits_file, "rb") as f:
        fold_splits = pickle.load(f)

    vt_ids: list[tuple[str, str]] = []
    with open(_CV_DIR / cfg.labels_file) as f:
        for line in f:
            parts = line.strip().split()
            vt_ids.append((parts[0], parts[1]))

    # collect all test (complex_id, variant) pairs not already cached
    # group by complex_id → list[variant]
    complex_to_variants: dict[str, list[str]] = {}
    for _, train_idx, test_idx in fold_splits:
        for idx in test_idx:
            complex_id, variant = vt_ids[idx]
            vt_id = f"{complex_id} {variant}"
            if vt_id not in cache:
                complex_to_variants.setdefault(complex_id, []).append(variant)

    # deduplicate variants per complex
    for cid in complex_to_variants:
        complex_to_variants[cid] = list(dict.fromkeys(complex_to_variants[cid]))

    session = requests.Session()

    # ── Phase 1: submit new jobs ──────────────────────────────────────────────
    already_pending_complexes = {j["complex_id"] for j in pending_jobs}
    n_submitted = 0
    for complex_id, variants in sorted(complex_to_variants.items()):
        if complex_id in already_pending_complexes:
            continue
        pdb_path = resolve_pdb(complex_id, cfg, u2g)
        if not pdb_path.exists():
            print(f"  MISSING PDB: {complex_id} → {pdb_path.name}", flush=True)
            for v in variants:
                cache[f"{complex_id} {v}"] = {"pred": float("nan"), "outcome": -1}
            continue

        # API list format: "{chain} {WT}{pos_1based}{MT}" e.g. "A S100P"
        # Variants in labels file use 0-indexed positions, so add 1 for PDB residue number.
        muts = [_variant_to_api_mut(v) for v in variants]
        batches = [muts[i:i + args.max_per_job]
                   for i in range(0, len(muts), args.max_per_job)]

        for batch in batches:
            job_id = _submit_list_job(pdb_path, batch, session)
            if job_id is None:
                for v in batch:
                    cache[f"{complex_id} {v}"] = {"pred": float("nan"), "outcome": -1}
                continue
            pending_jobs.append({
                "job_id":     job_id,
                "complex_id": complex_id,
                "variants":   batch,
            })
            n_submitted += 1
            print(f"  submitted job {job_id} ({complex_id}, {len(batch)} variants)",
                  flush=True)

    if n_submitted > 0:
        with open(jobs_path, "w") as f:
            json.dump(pending_jobs, f)
        print(f"Submitted {n_submitted} new jobs. Polling...", flush=True)

    # ── Phase 2: poll pending jobs ────────────────────────────────────────────
    backoff = args.poll_interval
    while pending_jobs:
        still_pending = []
        for job in pending_jobs:
            result = _poll_job(job["job_id"], session)
            if result is None:
                still_pending.append(job)
                continue

            # job done — parse predictions
            preds = _parse_result(result)
            for variant in job["variants"]:
                vt_id = f"{job['complex_id']} {variant}"
                entry = preds.get(variant)
                if entry is not None:
                    cache[vt_id] = entry
                else:
                    print(f"  WARNING: no prediction for {vt_id} in job {job['job_id']}",
                          flush=True)
                    cache[vt_id] = {"pred": float("nan"), "outcome": -1}

            print(f"  job {job['job_id']} done ({job['complex_id']})", flush=True)
            backoff = args.poll_interval  # reset on success

        # save cache and remaining jobs after each poll round
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        pending_jobs = still_pending
        with open(jobs_path, "w") as f:
            json.dump(pending_jobs, f)

        if still_pending:
            print(f"  {len(still_pending)} jobs still running — "
                  f"sleeping {backoff}s...", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)  # cap at 5 min

    # ── Phase 3: build flat pred arrays ──────────────────────────────────────
    all_preds:  list[float] = []
    all_binary: list[int]   = []
    n_missing = 0
    for _, train_idx, test_idx in fold_splits:
        for idx in test_idx:
            complex_id, variant = vt_ids[idx]
            vt_id = f"{complex_id} {variant}"
            entry = cache.get(vt_id)
            if entry is not None:
                all_preds.append(entry["pred"])
                all_binary.append(entry["outcome"])
            else:
                print(f"  WARNING: no cache entry for {vt_id}", flush=True)
                all_preds.append(float("nan"))
                all_binary.append(-1)
                n_missing += 1

    binary_npy = outdir / f"{cfg.name}_DDMutPPI_binary_labels.npy"
    print(f"\n{'='*50}", flush=True)
    print(f"Done: {len(all_preds)} predictions  {n_missing} missing", flush=True)
    np.save(out_npy,    np.array(all_preds,  dtype=np.float32))
    np.save(binary_npy, np.array(all_binary, dtype=np.int8))
    print(f"Saved: {out_npy}  shape={np.array(all_preds).shape}", flush=True)
    print(f"Saved: {binary_npy}", flush=True)
    print(f"Cache: {cache_path}  ({len(cache)} entries)", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DDMut-PPI API inference for all interaction-loss datasets"
    )
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    p.add_argument("--outdir", default=".",
                   help="Output directory for pred arrays and cache (default: current dir)")
    p.add_argument("--poll-interval", type=int, default=30,
                   help="Seconds between job status polls (default: 30)")
    p.add_argument("--max-per-job", type=int, default=500,
                   help="Max mutations per API list job (default: 500)")
    p.add_argument("--overwrite", action="store_true",
                   help="Ignore existing output .npy (but still reuse cache)")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
