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
                        Default: $MUTPRED_DATA_ROOT/mutpred_ppi_data
    MUTPRED_PPI_RESULTS_DIR
                        Generated figure/table artifacts (GCV pickles, blind-test
                        arrays, variant-DB predictions). Every results/<subdir>
                        constant below derives from this one root, so a sandbox
                        run (e.g. the reproduction notebook's QUICK mode) can
                        redirect all of them at once without touching the
                        canonical results/ tree that figures/*.png symlink into.
                        Default: $REPO_ROOT/results
    MUTPRED_CDHIT       Path to the cd-hit binary. Default: found on PATH.

Usage:
    from paths import REPO_ROOT, DATA_ROOT, WEIGHTS_DIR, cdhit_binary
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

def _env_path(var: str, default: Path) -> Path:
    val = os.environ.get(var)
    return Path(val).expanduser().resolve() if val else default


# ── repository-internal (never configurable; derived from this file) ──────────

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
WEIGHTS_DIR = REPO_ROOT / "weights"
DATASETS_DIR = REPO_ROOT / "datasets"
DATA_CACHES_DIR = REPO_ROOT / "data_caches"
# Overridable so a sandbox run (e.g. the reproduction notebook's QUICK mode) can
# write to results_quick/ without ever touching the canonical results/ tree
# that figures/*.png symlink into.
RESULTS_DIR = _env_path("MUTPRED_PPI_RESULTS_DIR", REPO_ROOT / "results")
FIGURES_DIR = REPO_ROOT / "figures"

# ── data preparation tiers (see docs/DATA_PREPARATION.md) ────────────────────
#
# Raw published inputs to the mapping notebook. Not vendored and not in the
# Zenodo deposit -- obtain them from the original publications, see
# docs/DATA_SOURCES.md. SOURCE_DATA_RESTRICTED_DIR holds the four VarChAMP/IGVF
# files, which are unpublished and cannot be redistributed at all.
SOURCE_DATA_DIR = DATASETS_DIR / "source_data"
SOURCE_DATA_RESTRICTED_DIR = DATASETS_DIR / "source_data_restricted"

# Output of notebooks/map_ppi_datasets.py: the mapped-but-not-yet-split
# CSVs, the non-deduplicated master, the QC/audit trail, and the UniProt REST
# cache that makes a re-run offline. Env-overridable for anyone keeping this
# tree elsewhere; the in-repo location is the real one.
MAPPING_DIR = _env_path("MUTPRED_PPI_MAPPING_DIR", DATASETS_DIR / "source_mapping")

# The prepared train/eval data layer: validated row tables, GCV splits,
# sequences, the contact-graph store and the per-method embedding caches.
# Named for its domain, matching its VARIANT_DBS sibling below, rather than
# for the mapping generation that produced it -- "mapped090826" was two
# characters from "mapping090826" while meaning something quite different.
TRAINING_EVAL_DIR = DATASETS_DIR / "training_eval"

# One results/ tree (2026-09-10): results_revisions/ is gone, its subdirectories
# moved under results/ unchanged except the three renamed below. Every script
# imports these constants rather than hardcoding "results/<subdir>", so a
# future rename is a one-line change here instead of a repo-wide grep.
GCV_RESULTS_DIR = RESULTS_DIR / "gcv"                          # was macro_aucs/
ROBUSTNESS_DIR = RESULTS_DIR / "robustness"                    # was robustness_analyses/
PROTEIN_CLASS_DIR = RESULTS_DIR / "protein_class"              # was protein_class_enrichment/
VARCHAMP_BLIND_TEST_DIR = RESULTS_DIR / "varchamp_seqcnf_newvar_eval"
# All variant-repository predictions MUST come from the single all-data
# model (weights/MutPred-PPI.pt, trained on sahni_fragoza_varchamp_all_mapped090826).
# There must be only one variant_dbs_* results tree; a second one scored by any
# other model (SF-only, a fold ensemble, ...) is a correctness bug, not a valid
# alternative -- see run_variant_db_inference.py::assert_all_data_model.
VARIANT_DBS_DIR = RESULTS_DIR / "variant_dbs_all_data"
VARIANT_DBS_STABILITY_DIR = RESULTS_DIR / "variant_dbs_stability"
VARIANT_DBS_CLASSIFIED_DIR = RESULTS_DIR / "variant_dbs_classified"
STABILITY_INTERACTION_DIR = RESULTS_DIR / "stability_interaction"
COSMIC_STAT_TEST_DIR = RESULTS_DIR / "cosmic_stat_test"
BICLASS_GCV_DIR = RESULTS_DIR / "biclass_gcv"
DATASET_COMPARISON_DIR = RESULTS_DIR / "dataset_comparison"
MASTER_VARIANT_DB_CSV = RESULTS_DIR / "master_variant_db_predictions.csv.gz"

# Small annotation/label inputs copied into the repo so every analysis script
# resolves inside the tree.  Delivered via the Zenodo bundle (datasets/ is
# gitignored); see docs/DATA_SOURCES.md.
ANNOTATIONS_DIR = DATASETS_DIR / "annotations"
# COSMIC/HGMD-derived summaries.  Same role, but licence-restricted: excluded
# from the Zenodo deposit and from git.  Analyses that need these must degrade
# with a clear message when they are absent rather than fail obscurely.
ANNOTATIONS_LICENSED_DIR = DATASETS_DIR / "annotations_licensed"
ESIGNET_SUPPLEMENTS_DIR = DATASETS_DIR / "esignet_supplements"

# Symlinks to large, machine-local trees that cannot ship (unpublished VarChAMP,
# multi-GB embedding/graph sets).  Created by scripts/link_external.sh.
EXTERNAL_DIR = REPO_ROOT / "external"

# ── comparison-method upstream checkouts ─────────────────────────────────────
#
# ONE directory for every third-party method, gitignored, so a reader clones
# each upstream repository into a predictable place instead of hunting for the
# path each script happens to expect.  Before 2026-09-10 these were scattered
# across three different roots plus one hardcoded absolute path
# ($DATA_ROOT/2026/mint, $DATA_ROOT/2026/PPLM, external/esignet, and an
# absolute path to MutPPI), and SAAMBE-3D was vendored into
# src/ outright.
#
# See docs/REPRODUCING_ANALYSES.md for the repository URL and pinned commit of
# each method.
EXTERNAL_METHODS_DIR = _env_path("MUTPRED_PPI_METHODS_DIR",
                                 REPO_ROOT / "external_methods")

# name -> (subdirectory, upstream URL). The URL is only ever used to build an
# error message; nothing here clones anything for you.
EXTERNAL_METHODS = {
    "saambe3d": ("saambe3d", "http://compbio.clemson.edu/SAAMBE-3D/"),
    "mint":     ("mint",     "https://github.com/VarunUllanat/mint"),
    "pplm":     ("PPLM",     "https://github.com/ChengfeiYan/PPLM"),
    "esignet":  ("esignet",  "https://github.com/Liu-Jing/eSIG-Net"),
    "mutppi":   ("MutPPI",   "https://github.com/Wang-Lin-boop/MutPPI"),
}


def method_dir(name: str, required: bool = True) -> Path:
    """Path to one comparison method's upstream checkout.

    Raises with the clone command rather than letting a missing checkout surface
    later as an ImportError from deep inside a `sys.path` insert.
    """
    if name not in EXTERNAL_METHODS:
        raise KeyError(f"unknown method {name!r}; known: {sorted(EXTERNAL_METHODS)}")
    sub, url = EXTERNAL_METHODS[name]
    path = EXTERNAL_METHODS_DIR / sub
    if required and not path.is_dir():
        raise FileNotFoundError(
            f"{name} upstream checkout not found at {path}.\n"
            f"  Clone it there:\n"
            f"      mkdir -p {EXTERNAL_METHODS_DIR}\n"
            f"      git clone {url} {path}\n"
            f"  or set MUTPRED_PPI_METHODS_DIR to a directory that contains "
            f"'{sub}'.\n"
            f"  See docs/REPRODUCING_ANALYSES.md for the pinned commit.")
    return path


# ── external, configurable ────────────────────────────────────────────────────

DATA_ROOT = _env_path("MUTPRED_DATA_ROOT", REPO_ROOT.parent)


def _default_cv_dir() -> Path:
    """Canonical CV artifacts, in-repo by default.

    `datasets/cv_reference/` now holds the full set (orderings, clusters, fold
    splits, test classes, label tables, and the SKEMPI-method prediction arrays),
    so nothing resolves to an external machine-specific directory any more.
    An out-of-tree copy is still honoured when present, and MUTPRED_CV_DIR
    overrides both.
    """
    if (DATASETS_DIR / "cv_reference").is_dir():
        return DATASETS_DIR / "cv_reference"
    return DATA_ROOT / "cv_splits"


CV_DIR = _env_path("MUTPRED_CV_DIR", _default_cv_dir())
CACHE_DIR = _env_path("MUTPRED_CACHE_DIR", DATA_ROOT / "mutpred_ppi_data")

# Frequently used subtrees of DATA_ROOT.
HOME_DIR = DATA_ROOT / "home"
REVISIONS_DIR = DATA_ROOT / "2026"


def _sibling_conda_envs():
    """Every conda environment reachable from the running interpreter.

    Derived from `sys.prefix`, so it follows whatever conda installation is in
    use rather than assuming a location.
    """
    prefix = Path(sys.prefix).resolve()
    roots = []
    for base in (prefix.parent, prefix.parent.parent / "envs"):
        if base.is_dir():
            roots.extend(sorted(p for p in base.iterdir() if p.is_dir()))
    return roots


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
    # cd-hit is often installed into a *different* conda environment than the
    # one running this code (it is a bioconda binary, not a Python package), so
    # it is frequently absent from PATH here while present a directory away.
    # Search sibling environments rather than hardcoding one machine's install
    # -- src/ must contain no absolute paths, which the previous
    # `_LEGACY_CDHIT` constant violated (and leaked a username with it).
    for env_root in _sibling_conda_envs():
        candidate = env_root / "bin" / "cd-hit"
        if candidate.is_file():
            return str(candidate)
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
    """Every resolved path plus whether it exists — the setup self-test.

    docs/SETUP.md points readers here, so this must cover all three data tiers:
    in-repo (datasets/), symlinked (external/), and configurable roots.
    """
    def row(label: str, path, env: str = "") -> str:
        mark = "ok     " if Path(path).exists() else "MISSING"
        suffix = f"   ({env})" if env else ""
        return f"  {mark}  {label:<26} {path}{suffix}"

    lines = ["configurable roots:"]
    lines += [
        row("DATA_ROOT", DATA_ROOT, "MUTPRED_DATA_ROOT"),
        row("CV_DIR", CV_DIR, "MUTPRED_CV_DIR"),
        row("CACHE_DIR", CACHE_DIR, "MUTPRED_CACHE_DIR"),
        row("RESULTS_DIR", RESULTS_DIR, "MUTPRED_PPI_RESULTS_DIR"),
    ]
    lines += ["", "in-repo (Zenodo-delivered; datasets/ is gitignored):"]
    lines += [
        row("DATASETS_DIR", DATASETS_DIR),
        row("SOURCE_DATA_DIR", SOURCE_DATA_DIR),
        row("SOURCE_DATA_RESTRICTED_DIR", SOURCE_DATA_RESTRICTED_DIR),
        row("MAPPING_DIR", MAPPING_DIR, "MUTPRED_PPI_MAPPING_DIR"),
        row("TRAINING_EVAL_DIR", TRAINING_EVAL_DIR),
        row("ANNOTATIONS_DIR", ANNOTATIONS_DIR),
        row("ANNOTATIONS_LICENSED_DIR", ANNOTATIONS_LICENSED_DIR),
        row("ESIGNET_SUPPLEMENTS_DIR", ESIGNET_SUPPLEMENTS_DIR),
        row("WEIGHTS_DIR", WEIGHTS_DIR),
        row("GCV_RESULTS_DIR", GCV_RESULTS_DIR),
        row("VARCHAMP_BLIND_TEST_DIR", VARCHAMP_BLIND_TEST_DIR),
        row("ROBUSTNESS_DIR", ROBUSTNESS_DIR),
        row("PROTEIN_CLASS_DIR", PROTEIN_CLASS_DIR),
        row("VARIANT_DBS_DIR", VARIANT_DBS_DIR),
        row("VARIANT_DBS_STABILITY_DIR", VARIANT_DBS_STABILITY_DIR),
    ]
    lines += ["", "symlinked (run scripts/link_external.sh):"]
    lines += [row("EXTERNAL_DIR", EXTERNAL_DIR)]
    if EXTERNAL_DIR.is_dir():
        broken = sorted(p.name for p in EXTERNAL_DIR.iterdir() if not p.exists())
        lines += [f"  {'ok     ' if not broken else 'WARN   '}  "
                  f"{'links':<26} {len(list(EXTERNAL_DIR.iterdir()))} present, "
                  f"{len(broken)} broken"
                  + (f": {', '.join(broken)}" if broken else "")]
    lines += ["", "tools:"]
    try:
        lines += [row("cd-hit", cdhit_binary(), "MUTPRED_CDHIT")]
    except FileNotFoundError:
        lines += ["  MISSING  cd-hit                     not found "
                  "(conda install -c bioconda cd-hit; only needed to build splits "
                  "from scratch)"]
    return "\n".join([f"REPO_ROOT  {REPO_ROOT}", ""] + lines)


if __name__ == "__main__":
    # describe() already reports cd-hit; don't print it twice.
    print(describe())
