"""Central path resolution for the MutPred-PPI repository.

Every path used by the codebase resolves through here so that:
  * nothing hardcodes an absolute repo location (the repo is relocatable), and
  * every external data root can be redirected with an environment variable.

Environment variables (all optional; defaults suit the original workstation):

    MUTPRED_DATA_ROOT   Large shared data tree: contact graphs, ProtT5/ESM
                        embedding pickles, variant-database subsets.
                        Default: the repository's parent directory.
    MUTPRED_CV_DIR      Cross-validation reference artifacts (canonical vt_id
                        orderings, label text files). Fold splits themselves are
                        generated inline and no longer read from here.
                        Default: $MUTPRED_DATA_ROOT/cv_splits if present,
                        else the legacy workstation location.
    MUTPRED_CACHE_DIR   Large regenerable prediction/embedding caches
                        (mint_cache.pkl, pplm_cache.pkl, ...).
                        Default: $MUTPRED_DATA_ROOT/nm_revisions
    MUTPRED_CDHIT       Path to the cd-hit binary. Default: found on PATH.

Usage:
    from paths import REPO_ROOT, DATA_ROOT, WEIGHTS_DIR, cdhit_binary
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# ── repository-internal (never configurable; derived from this file) ──────────

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
WEIGHTS_DIR = REPO_ROOT / "weights"
DATASETS_DIR = REPO_ROOT / "datasets"
DATA_CACHES_DIR = REPO_ROOT / "data_caches"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_REV_DIR = REPO_ROOT / "results_revisions"
FIGURES_DIR = REPO_ROOT / "figures"

GCV_RESULTS_DIR = RESULTS_REV_DIR / "macro_aucs"
VCFP_RESULTS_DIR = RESULTS_DIR / "varchamp_seqcnf_newvar_eval"

_LEGACY_CV_DIR = Path("/home/rcstewart/gnn/ppi_interaction_loss/cv_splits")
_LEGACY_CDHIT = "/home/rcstewart/miniconda3/envs/pytorch_env/bin/cd-hit"


def _env_path(var: str, default: Path) -> Path:
    val = os.environ.get(var)
    return Path(val).expanduser().resolve() if val else default


# ── external, configurable ────────────────────────────────────────────────────

DATA_ROOT = _env_path("MUTPRED_DATA_ROOT", REPO_ROOT.parent)


def _default_cv_dir() -> Path:
    candidate = DATA_ROOT / "cv_splits"
    if candidate.is_dir():
        return candidate
    return _LEGACY_CV_DIR


CV_DIR = _env_path("MUTPRED_CV_DIR", _default_cv_dir())
CACHE_DIR = _env_path("MUTPRED_CACHE_DIR", DATA_ROOT / "nm_revisions")

# Frequently used subtrees of DATA_ROOT.
HOME_DIR = DATA_ROOT / "home"
REVISIONS_DIR = DATA_ROOT / "2026"


def cdhit_binary() -> str:
    """Locate the cd-hit executable.

    Raises with an actionable message rather than letting a missing binary
    surface later as an empty cluster list.
    """
    explicit = os.environ.get("MUTPRED_CDHIT")
    if explicit:
        if not Path(explicit).is_file():
            raise FileNotFoundError(f"MUTPRED_CDHIT is set to {explicit!r}, which does not exist")
        return explicit
    found = shutil.which("cd-hit")
    if found:
        return found
    if Path(_LEGACY_CDHIT).is_file():  # original workstation install
        return _LEGACY_CDHIT
    raise FileNotFoundError(
        "cd-hit not found on PATH. Install it (conda install -c bioconda cd-hit) "
        "or set MUTPRED_CDHIT to the binary. Sequence clustering supplies the "
        "GroupKFold groups, so cross-validation cannot run without it."
    )


CV_REFERENCE_DIR = DATASETS_DIR / "cv_reference"


def cv_reference_dir() -> Path:
    """Canonical CV reference artifacts (orderings, fold splits, test classes).

    Prefers the in-repo copy written by src/analysis/export_cv_reference.py,
    falling back to the legacy external cv_splits directory. Consumers must read
    these rather than the legacy files, whose seed-1 test classes are known to
    disagree with their own fold splits.
    """
    if CV_REFERENCE_DIR.is_dir() and any(CV_REFERENCE_DIR.iterdir()):
        return CV_REFERENCE_DIR
    return CV_DIR


def cache_file(name: str) -> Path:
    """Resolve a large regenerable cache (e.g. 'mint_cache.pkl').

    Prefers an in-repo data_caches/ copy, then MUTPRED_CACHE_DIR. Returns the
    CACHE_DIR candidate when neither exists so callers can report a clean
    'missing, regenerate with ...' error against a stable path.
    """
    for candidate in (DATA_CACHES_DIR / name, CACHE_DIR / name):
        if candidate.exists():
            return candidate
    return CACHE_DIR / name


def require(path: Path, what: str, hint: str = "") -> Path:
    """Fail early with context instead of deep inside a loader."""
    if not Path(path).exists():
        msg = f"{what} not found: {path}"
        if hint:
            msg += f"\n  {hint}"
        raise FileNotFoundError(msg)
    return Path(path)


def describe() -> str:
    return "\n".join([
        f"REPO_ROOT  {REPO_ROOT}",
        f"DATA_ROOT  {DATA_ROOT}   (MUTPRED_DATA_ROOT)",
        f"CV_DIR     {CV_DIR}   (MUTPRED_CV_DIR)",
        f"CACHE_DIR  {CACHE_DIR}   (MUTPRED_CACHE_DIR)",
    ])


if __name__ == "__main__":
    print(describe())
    try:
        print(f"cd-hit     {cdhit_binary()}")
    except FileNotFoundError as e:
        print(f"cd-hit     MISSING -- {e}")
