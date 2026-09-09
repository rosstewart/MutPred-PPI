#!/usr/bin/env python
"""DDMut-PPI inference for all 090826 canonical interaction-loss datasets.

Calls the DDMut-PPI REST API (https://biosig.lab.uq.edu.au/ddmut_ppi/api/)
to get ΔΔG predictions, with per-complex batching (≤100 mutations per job),
resume support, and exponential-backoff polling.

Structures are resolved by chain sequence, not by filename, matching the same
index used by MutPred-PPI and SAAMBE-3D. Mutation positions are 1-based (from
the canonical tables), so the API format "A {WT}{pos1}{MT}" requires no +1
adjustment. The interactor chain is identified from the structure index and
passed to the API.

Workflow:
  1. Load canonical rows + fold splits.
  2. Group all unique test variants by resolved PDB complex.
  3. Submit /api/list jobs (one job per complex, ≤100 mutations each).
  4. Poll all jobs until DONE; parse predictions into a cache dict.
  5. Build flat pred array in fold-test order; save as {dataset}_DDMutPPI_preds.npy.

Resume: if {outdir}/DDMutPPI_{dataset}_cache.pkl exists, already-computed
variants are skipped and their job_ids (in {outdir}/DDMutPPI_{dataset}_jobs.json)
are re-polled if still pending.

Per-job timeout: if a job has been RUNNING for more than --job-timeout seconds
(default 1200), it is resubmitted with the same PDB and mutations. After 5
resubmissions without success, all variants in that job are marked NaN.

Usage:
    conda run -n ppi python src/evaluation/ddmutppi_cv.py \\
        --dataset sahni_fragoza_mapped090826 --outdir ./results/
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import requests

_HERE = Path(__file__).resolve().parent

from evaluation.gcv_common import DATASET_CONFIGS, load_data, load_splits  # noqa: E402
from evaluation.graphs import GraphResolver, PDBResolver  # noqa: E402

_API_BASE = "https://biosig.lab.uq.edu.au/ddmut_ppi/api"


def _mut_to_api(mutation: str, chain: str) -> str:
    """Convert 1-based canonical mutation (e.g. 'S100P') to API format '{chain} S100P'."""
    return f"{chain} {mutation}"


def _api_key_to_mutation(api_key: str) -> str:
    """Convert API result key (e.g. 'A_S100P') back to canonical mutation 'S100P'."""
    return api_key.split("_", 1)[-1] if "_" in api_key else api_key


def _submit_list_job(pdb_path: Path, mutations: list[str],
                     session: requests.Session) -> str | None:
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


def _submit_with_retry(pdb_path: Path, mutations: list[str],
                       session: requests.Session, max_attempts: int = 3) -> str | None:
    delays = [5, 15, 45]
    for attempt in range(max_attempts):
        job_id = _submit_list_job(pdb_path, mutations, session)
        if job_id is not None:
            return job_id
        if attempt < max_attempts - 1:
            wait = delays[attempt]
            print(f"  Retrying submit in {wait}s (attempt {attempt + 2}/{max_attempts})…",
                  flush=True)
            time.sleep(wait)
    return None


def _poll_job(job_id: str, session: requests.Session) -> dict | None:
    try:
        resp = session.get(f"{_API_BASE}/list", data={"job_id": job_id}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("status") == "DONE":
            return data
        if isinstance(data, dict) and data.get("message") == "RUNNING":
            return None
        if isinstance(data, dict) and any(k not in ("job_id", "status", "results_page")
                                           for k in data):
            return data
        return None
    except Exception as exc:
        print(f"  POLL ERROR job {job_id}: {exc}", flush=True)
        return None


def _parse_result(result: dict) -> dict[str, dict]:
    """Extract {mutation (1-based) → {"pred": ddG, "outcome": binary_int}}."""
    _DISRUPTIVE = {"disruptive", "destabilizing", "increasing"}
    parsed: dict[str, dict] = {}
    for key, val in result.items():
        if key in ("job_id", "status", "results_page"):
            continue
        if isinstance(val, dict) and "prediction" in val:
            mut_key = _api_key_to_mutation(key)
            outcome_str = str(val.get("outcome", "")).lower()
            binary = (1 if any(kw in outcome_str for kw in _DISRUPTIVE)
                      else (0 if outcome_str else -1))
            parsed[mut_key] = {"pred": float(val["prediction"]), "outcome": binary}
    return parsed


def run(args: argparse.Namespace) -> None:
    cfg    = DATASET_CONFIGS[args.dataset]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cache_path = outdir / f"DDMutPPI_{cfg.name}_cache.pkl"
    jobs_path  = outdir / f"DDMutPPI_{cfg.name}_jobs.json"
    out_npy    = outdir / f"{cfg.name}_DDMutPPI_preds.npy"
    binary_npy = outdir / f"{cfg.name}_DDMutPPI_binary_labels.npy"

    if out_npy.exists() and binary_npy.exists() and not args.overwrite:
        print(f"Output already exists: {out_npy}  (use --overwrite to rerun)", flush=True)
        return

    # Cache key: "{interactor}_{partner} {mutation}" (1-based mutation, UniProt accessions)
    cache: dict[str, dict] = {}
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            raw = pickle.load(f)
        for k, v in raw.items():
            cache[k] = v if isinstance(v, dict) else {"pred": v, "outcome": -1}
        n_before = len(cache)
        if args.retry_nans:
            cache = {k: v for k, v in cache.items()
                     if not (isinstance(v, dict) and np.isnan(v.get("pred", 0.0)))}
            print(f"Loaded cache: {n_before} entries, removed {n_before - len(cache)} NaN "
                  "entries for retry", flush=True)
        else:
            print(f"Loaded cache: {len(cache)} variants already computed", flush=True)

    pending_jobs: list[dict] = []
    if jobs_path.exists():
        with open(jobs_path) as f:
            pending_jobs = json.load(f)
        now = time.time()
        for job in pending_jobs:
            job.setdefault("submitted_at", now)
            job.setdefault("n_resubmits", 0)
        print(f"Loaded {len(pending_jobs)} pending jobs from {jobs_path}", flush=True)

    rows = load_data(cfg)
    fold_splits, _ = load_splits(cfg, seed=args.seed)
    resolver = GraphResolver()
    pdbs = PDBResolver()

    print(f"Dataset: {cfg.name}  rows: {len(rows)}", flush=True)

    # Group all unique test (interactor, partner, mutation) triples not yet cached.
    # Key: (interactor, partner) → list[mutation]; row_key: "{interactor}_{partner} {mutation}"
    complex_to_mutations: dict[tuple[str, str], list[str]] = {}
    complex_to_pdb: dict[tuple[str, str], tuple[Path | None, str]] = {}

    for _, _train_idx, test_idx in fold_splits:
        for idx in test_idx:
            row = rows.loc[idx]
            key = (row["interactor"], row["partner"])
            row_key = f"{row['interactor']}_{row['partner']} {row['mutation']}"
            if row_key in cache:
                continue
            if key not in complex_to_pdb:
                hit = resolver.find(row["interactor_sequence"], row["partner_sequence"])
                if hit is None:
                    complex_to_pdb[key] = (None, "A")
                else:
                    mat_path, interactor_is_first, _ = hit
                    complex_to_pdb[key] = (pdbs.find(mat_path),
                                           "A" if interactor_is_first else "B")
            complex_to_mutations.setdefault(key, []).append(row["mutation"])

    # Deduplicate mutations per complex
    for k in complex_to_mutations:
        complex_to_mutations[k] = list(dict.fromkeys(complex_to_mutations[k]))

    session = requests.Session()

    # ── Phase 1: submit new jobs ──────────────────────────────────────────────
    already_pending = {j["complex_key"] for j in pending_jobs}
    n_submitted = 0
    for (interactor, partner), mutations in sorted(complex_to_mutations.items()):
        complex_key = f"{interactor}_{partner}"
        if complex_key in already_pending:
            continue
        pdb_path, chain = complex_to_pdb[(interactor, partner)]
        if pdb_path is None:
            print(f"  MISSING PDB: {interactor}/{partner}", flush=True)
            for mut in mutations:
                cache[f"{interactor}_{partner} {mut}"] = {"pred": float("nan"), "outcome": -1}
            continue

        api_muts = [_mut_to_api(mut, chain) for mut in mutations]
        batches = [api_muts[i:i + args.max_per_job]
                   for i in range(0, len(api_muts), args.max_per_job)]
        mut_batches = [mutations[i:i + args.max_per_job]
                       for i in range(0, len(mutations), args.max_per_job)]

        for api_batch, mut_batch in zip(batches, mut_batches):
            job_id = _submit_with_retry(pdb_path, api_batch, session)
            if job_id is None:
                for mut in mut_batch:
                    cache[f"{interactor}_{partner} {mut}"] = {"pred": float("nan"), "outcome": -1}
                continue
            pending_jobs.append({
                "job_id":       job_id,
                "complex_key":  complex_key,
                "interactor":   interactor,
                "partner":      partner,
                "mutations":    mut_batch,
                "submitted_at": time.time(),
                "n_resubmits":  0,
            })
            n_submitted += 1
            print(f"  submitted job {job_id} ({interactor}/{partner}, "
                  f"{len(mut_batch)} mutations)", flush=True)

    if n_submitted > 0:
        with open(jobs_path, "w") as f:
            json.dump(pending_jobs, f)
        print(f"Submitted {n_submitted} new jobs. Polling...", flush=True)

    # ── Phase 2: poll pending jobs ────────────────────────────────────────────
    backoff = args.poll_interval
    while pending_jobs:
        still_pending = []
        for job in pending_jobs:
            elapsed = time.time() - job.get("submitted_at", 0)
            if elapsed > args.job_timeout:
                n_rsub = job.get("n_resubmits", 0)
                if n_rsub >= 5:
                    print(f"  [WARN] job {job['job_id']} ({job['complex_key']}) timed out "
                          f"after 5 resubmits; marking NaN", flush=True)
                    for mut in job["mutations"]:
                        cache[f"{job['interactor']}_{job['partner']} {mut}"] = {
                            "pred": float("nan"), "outcome": -1}
                    continue
                print(f"  [WARN] job {job['job_id']} timed out after {elapsed:.0f}s; "
                      f"resubmitting (attempt {n_rsub + 1}/5)…", flush=True)
                pdb_path, chain = complex_to_pdb[(job["interactor"], job["partner"])]
                api_muts = [_mut_to_api(m, chain) for m in job["mutations"]]
                new_id = _submit_with_retry(pdb_path, api_muts, session)
                if new_id is None:
                    still_pending.append(job)
                else:
                    job["job_id"]       = new_id
                    job["submitted_at"] = time.time()
                    job["n_resubmits"]  = n_rsub + 1
                    print(f"  Resubmitted as {new_id} (#{job['n_resubmits']})", flush=True)
                    still_pending.append(job)
                continue

            result = _poll_job(job["job_id"], session)
            if result is None:
                still_pending.append(job)
                continue

            preds = _parse_result(result)
            for mut in job["mutations"]:
                row_key = f"{job['interactor']}_{job['partner']} {mut}"
                entry = preds.get(mut)
                if entry is not None:
                    cache[row_key] = entry
                else:
                    print(f"  WARNING: no prediction for {row_key} in job {job['job_id']}",
                          flush=True)
                    cache[row_key] = {"pred": float("nan"), "outcome": -1}
            print(f"  job {job['job_id']} done ({job['complex_key']})", flush=True)
            backoff = args.poll_interval

        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        pending_jobs = still_pending
        with open(jobs_path, "w") as f:
            json.dump(pending_jobs, f)

        if still_pending:
            print(f"  {len(still_pending)} jobs still running — "
                  f"sleeping {backoff}s...", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)

    # ── Phase 3: build flat pred arrays ──────────────────────────────────────
    all_preds:  list[float] = []
    all_binary: list[int]   = []
    n_missing = 0
    for _, _train_idx, test_idx in fold_splits:
        for idx in test_idx:
            row = rows.loc[idx]
            row_key = f"{row['interactor']}_{row['partner']} {row['mutation']}"
            entry = cache.get(row_key)
            if entry is not None:
                all_preds.append(entry["pred"])
                all_binary.append(entry["outcome"])
            else:
                print(f"  WARNING: no cache entry for {row_key}", flush=True)
                all_preds.append(float("nan"))
                all_binary.append(-1)
                n_missing += 1

    print(f"\n{'='*50}", flush=True)
    print(f"Done: {len(all_preds)} predictions  {n_missing} missing", flush=True)
    np.save(out_npy,    np.array(all_preds,  dtype=np.float32))
    np.save(binary_npy, np.array(all_binary, dtype=np.int8))
    print(f"Saved: {out_npy}  shape={np.array(all_preds).shape}", flush=True)
    print(f"Saved: {binary_npy}", flush=True)
    print(f"Cache: {cache_path}  ({len(cache)} entries)", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DDMut-PPI API inference — 090826 canonical datasets")
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    p.add_argument("--seed", type=int, default=0,
                   help="GCV split seed (default: 0)")
    p.add_argument("--outdir", default=".", help="Output directory for pred arrays and cache")
    p.add_argument("--poll-interval", type=int, default=30,
                   help="Seconds between job status polls (default: 30)")
    p.add_argument("--max-per-job", type=int, default=100,
                   help="Max mutations per API list job (default: 100)")
    p.add_argument("--job-timeout", type=int, default=1200,
                   help="Seconds before a RUNNING job is resubmitted (default: 1200)")
    p.add_argument("--overwrite", action="store_true",
                   help="Ignore existing output .npy (but still reuse cache)")
    p.add_argument("--retry-nans", action="store_true",
                   help="Remove NaN cache entries so they are resubmitted to the API")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
