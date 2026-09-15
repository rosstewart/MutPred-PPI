#!/usr/bin/env python
"""Reject inputs from the pre-090826 pipeline instead of silently accepting them.

The repo was rebaselined on 2026-09-08/09: every canonical dataset now carries
a `_mapped090826` suffix (`sahni_only_mapped090826`, `fragoza_only_mapped090826`,
`sahni_fragoza_mapped090826`, `varchamp_all_mapped090826`,
`sahni_fragoza_varchamp_all_mapped090826` -- see `utils.gcv_common.DATASET_CONFIGS`).
Everything that predates that rebaseline -- `.mat`/`.pos`/`.neg`/`.labels`/`.vt_ids`
files, `all_vt_ids_and_labels.txt`, `training_data_internal.csv`,
`datasets/sfvcfp_rows.csv.gz`, the retired `sahni_varchamp1p_cava` /
`sahni_fragoza_varchamp2026` / `sahni_fragoza_varchamp_full_pooled` dataset
name family, and any path outside `REPO_ROOT`/`DATA_ROOT` -- is obsolete and
must be regenerated, never read.

`reject_legacy` raises rather than warns: a warning is easy to miss in a long
GPU job's log; a raised `LegacyInputError` stops the run before it produces a
number keyed to the wrong data.
"""
from __future__ import annotations

import re
from pathlib import Path

from paths import DATA_ROOT, EXTERNAL_METHODS_DIR, REPO_ROOT

# Extensions from the retired per-source pipeline (FASTA/label-file era).
# The one live mapping generation. Every canonical dataset name ends with this,
# and every consumer that builds a filename from a dataset should use it rather
# than writing the literal -- so that when the mapping is regenerated, changing
# this constant makes stale files fail to resolve instead of being picked up.
#
# Deliberately NOT a fallback: code that accepts both a suffixed and an
# unsuffixed filename will silently read whichever exists, which is how stale
# arrays get into a figure.
DATASET_SUFFIX = "_mapped090826"


_LEGACY_SUFFIXES = {".mat", ".pos", ".neg", ".labels", ".vt_ids"}

# Filename substrings that identify a retired artifact even when the
# extension looks innocuous (e.g. a `.txt` label file or a `.csv.gz` table
# built by a since-archived script).
_LEGACY_NAME_PATTERNS = [
    re.compile(r"all_vt_ids"),
    re.compile(r"_and_labels\.txt$"),
    re.compile(r"^training_data\.csv$"),
    re.compile(r"^training_data_internal\.csv$"),
    re.compile(r"^sfvcfp_rows\.csv\.gz$"),
    re.compile(r"(?<![a-z])fold_splits\.pkl$"),  # unsuffixed, pre-090826 only
]

# Retired dataset-name family. Any of these appearing in a path/filename means
# the artifact was built under a naming scheme this repo no longer uses --
# distinct from (and in addition to) the five live `*_mapped090826` configs.
_LEGACY_DATASET_TOKENS = [
    "varchamp1p_cava",
    "varchamp2026",
    "varchamp_full_pooled",
    "varchamp_pooled",
    "varchamp_full",
    "sfvcfp",
    "vcfp",
]

# Freshness floor: the 090826 canonical-table rebuild. Anything under
# REPO_ROOT/DATA_ROOT older than this predates the rebaseline.
_REBASELINE_TS = 1757304000.0  # 2026-09-08T00:00:00, local


class LegacyInputError(RuntimeError):
    """Raised when a resolved input is from the retired pre-090826 pipeline."""


def _matches_legacy_name(name: str) -> str | None:
    if any(name.endswith(suf) for suf in _LEGACY_SUFFIXES):
        return f"retired extension {Path(name).suffix!r}"
    for pat in _LEGACY_NAME_PATTERNS:
        if pat.search(name):
            return f"retired filename pattern {pat.pattern!r}"
    lowered = name.lower()
    for token in _LEGACY_DATASET_TOKENS:
        if token in lowered:
            return f"retired dataset-name token {token!r}"
    return None



def _under(path: Path, root: Path) -> bool:
    """True if `path` is `root` or sits beneath it."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False

def reject_legacy(*paths: str | Path, check_mtime: bool = True) -> None:
    """Raise `LegacyInputError` if any path looks like a pre-090826 artifact.

    Checks, in order: retired extension/filename pattern, retired
    dataset-name token, location outside the repo/data tree, and (when
    `check_mtime` and the path exists) an mtime before the 090826 rebuild.
    Nonexistent paths are still checked for name/location -- a caller often
    resolves a legacy path that was since deleted, and the point is to catch
    the *reference*, not just a file that happens to still be on disk.
    """
    for raw in paths:
        p = Path(raw)
        reason = _matches_legacy_name(p.name)
        if reason:
            raise LegacyInputError(f"{p}: {reason} -- regenerate from datasets/training_eval/")

        # Third root: the comparison-method checkouts. `MUTPRED_PPI_METHODS_DIR`
        # is a documented setting, and `external_methods/<method>` is commonly a
        # symlink to wherever that upstream repo was cloned, so a pinned
        # checkpoint legitimately resolves anywhere on the filesystem. Those
        # directories hold third-party code and weights, never our data, so
        # nothing this guard looks for can be inside one.
        #
        # Matched on the LOGICAL path rather than the resolved one: the caller
        # built it from `paths.method_dir()`, and resolving would follow the
        # per-method symlink out to an arbitrary tree. Both forms are tried, so
        # it holds whether or not the checkout is a link.
        # Both the resolved and the logical (symlinks unfollowed) form are
        # tried against every root. What this check is for is a hardcoded
        # reference to a retired external script directory, and such a path sits
        # outside the tree in BOTH forms. A multi-GB embedding cache symlinked
        # into datasets/ from other storage, or an external_methods/<method>
        # checkout that is a link, is logically inside and must pass -- matching
        # only the resolved form rejected exactly those.
        resolved = p.resolve()
        logical = p if p.is_absolute() else Path.cwd() / p
        roots = (REPO_ROOT, DATA_ROOT, EXTERNAL_METHODS_DIR,
                 EXTERNAL_METHODS_DIR.resolve())
        in_tree = any(
            _under(candidate, root)
            for candidate in (resolved, logical)
            for root in roots
        )
        if not in_tree:
            raise LegacyInputError(
                f"{p}: resolves outside REPO_ROOT ({REPO_ROOT}), DATA_ROOT "
                f"({DATA_ROOT}) and the method checkouts "
                f"({EXTERNAL_METHODS_DIR}) -- this is almost always a hardcoded "
                f"path to a retired external script directory (e.g. ~/gnn/..., "
                f"~/ppi_lossgain/2026/...); point it at the canonical tree instead"
            )

        if check_mtime and resolved.exists() and resolved.is_file():
            if resolved.stat().st_mtime < _REBASELINE_TS:
                raise LegacyInputError(
                    f"{p}: predates the 2026-09-08 canonical rebuild "
                    f"(mtime {resolved.stat().st_mtime}) -- regenerate from "
                    f"datasets/training_eval/"
                )


def reject_legacy_dataset_name(name: str) -> None:
    """Raise if `name` names a retired dataset (not one of the five *_mapped090826 configs)."""
    lowered = name.lower()
    for token in _LEGACY_DATASET_TOKENS:
        if token in lowered:
            raise LegacyInputError(
                f"{name!r}: retired dataset-name token {token!r} -- only the five "
                f"*_mapped090826 configs in utils.gcv_common.DATASET_CONFIGS are live"
            )
