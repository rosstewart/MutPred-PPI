"""The published stratum tables must obey the coverage rule Methods states.

Two independent guards, both for bugs that reached published figures.

1. THE COVERAGE RULE WAS IMPLEMENTED AND NEVER SWITCHED ON. `build_arrays`
   drops a variant when BioGRID lists at least `min_partners` partners for it
   but fewer were scored. `--min-partners` defaulted to 1, the notebook never
   overrode it, and the only caller passing 3 was a unit test. 6,331 variants
   that the rule would have removed sat in the shipped tables, so the artifacts
   disagreed with the Methods sentence. `test_edgotypes.py` checks the rule's
   logic on synthetic input; this checks that it actually ran.

2. A SINGLE-LABEL DATABASE MUST NOT BE FILTERED THROUGH A FOLDING-CAP PICKLE.
   `*_variant_subset.pkl` files are written by the mapping scripts to cap the
   AlphaFold budget (<=10 partners per protein, stopping at 600 complexes over
   an unordered set). They are not variant lists. Using one as a group
   definition has now happened twice -- the ASD cohort, then HGMD, where it
   discarded 9,822 of 13,383 scored variants and 91% of the AR ones, producing
   the AR:AD ratio of 1:11.3 in Fig 5 against ClinVar's 1:3.06. Nothing
   detected either.
"""
import ast
import glob
from pathlib import Path

import pandas as pd
import pytest

from paths import REPO_ROOT, VARIANT_DBS_DIR

#: The rule as stated in Methods.
MIN_PARTNERS = 3

#: Databases with one label for every row, so there is no sub-group to select
#: and no legitimate reason to consult a subset file.
SINGLE_LABEL_DBS = ("hgmd", "asd")

CLASSIFY_SRC = REPO_ROOT / "src" / "analysis" / "classify_variant_dbs.py"


def _tables() -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(VARIANT_DBS_DIR / "*" / "*.csv.gz")))


needs_data = pytest.mark.skipif(not _tables(),
                                reason="no classified stratum tables on disk")


@needs_data
def test_no_variant_violates_the_coverage_rule():
    """No stratum may hold a variant BioGRID knows >=3 partners for, scored on <3.

    Reported in one message rather than per file: the failure is a pipeline-wide
    property of the run that produced the tables, so naming every affected
    stratum is more useful than stopping at the first.
    """
    offenders = {}
    for path in _tables():
        df = pd.read_csv(path)
        if "n_biogrid_partners" not in df.columns:
            continue
        per_variant = df.groupby(["uniprot", "variant"]).agg(
            n_scored=("partner", "nunique"),
            n_biogrid=("n_biogrid_partners", "max"))
        bad = int(((per_variant.n_biogrid >= MIN_PARTNERS)
                   & (per_variant.n_scored < MIN_PARTNERS)).sum())
        if bad:
            offenders[f"{path.parent.name}/{path.stem}"] = bad

    assert not offenders, (
        f"{sum(offenders.values())} variants violate the >={MIN_PARTNERS}-tested-"
        f"partners rule: {offenders}. Re-run classify_variant_dbs.py with "
        f"--min-partners {MIN_PARTNERS}.")


def test_single_label_databases_consult_no_subset_file():
    """`process_hgmd` and `process_asd` must not open a `*_variant_subset.pkl`.

    Read from the source rather than from behaviour: the artifact these produce
    looks entirely plausible when the wrong file is used, which is why the bug
    survived two figures.
    """
    tree = ast.parse(CLASSIFY_SRC.read_text())
    funcs = {node.name: node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef)}

    for db in SINGLE_LABEL_DBS:
        name = f"process_{db}"
        assert name in funcs, f"{name} not found in {CLASSIFY_SRC}"
        literals = [n.value for n in ast.walk(funcs[name])
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        # The docstrings explain the bug and so name the file on purpose.
        code_literals = literals[1:] if ast.get_docstring(funcs[name]) else literals
        offending = [s for s in code_literals if "variant_subset" in s]
        assert not offending, (
            f"{name} references {offending}; every row of {db} carries the same "
            f"label, so a subset file can only be the folding-budget cap")
