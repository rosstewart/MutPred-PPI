"""The mapping must never resolve a NEW ambiguous GeneID by a length heuristic.

Background (2026-09-10). `datasets/training_eval/` was found to encode three wrong
protein assignments. The cause was not a missing reviewed/human filter -- both are
enforced, and every ambiguous candidate is `reviewed` + `9606`. UniProt genuinely
cross-references one NCBI GeneID to several distinct reviewed human entries, and
the mapping was breaking those ties on accession string order, then on sequence
length. Both are proxies, applied silently.

For `51207` the two candidates are DUSP13A (Q6B8I1, 188 aa) and DUSP13B (Q9UII6,
198 aa), which share **3.7% identity** -- so the wrong pick does not perturb a
sequence, it substitutes an unrelated protein.

The rule these tests pin: ambiguity whose candidates are sequence-IDENTICAL is
harmless and passes; ambiguity whose candidates DIVERGE must be pinned explicitly
in `ENTREZ_AMBIGUOUS_PINS`, or the mapping raises.
"""
import re
from pathlib import Path

import pandas as pd
import pytest

_NB = Path("notebooks/map_ppi_datasets.py")
_AMB = Path("datasets/source_mapping/intermediate_files/ambiguous_entrez_ids.csv")

# The three GeneIDs with genuinely different products, and the entry each must
# resolve to. Hard-coded on purpose: this is the record of the decision.
EXPECTED_PINS = {
    "51207": "Q9UII6",   # DUSP13B, not DUSP13A -- 3.7% identity between them
    "83871": "Q9BZG1",   # Rab-34 itself, not the NARR readthrough P0DI83
    "9465":  "Q9P0M2",   # AKAP7 gamma (348 aa), not alpha (104 aa)
}


def _pins_from_notebook():
    src = _NB.read_text()
    block = re.search(r"ENTREZ_AMBIGUOUS_PINS = \{(.*?)\n\}", src, re.S)
    assert block, "ENTREZ_AMBIGUOUS_PINS not found in the mapping notebook"
    return dict(re.findall(r'"(\d+)":\s*"([A-Z0-9]+)"', block.group(1)))


def test_notebook_pins_the_three_divergent_geneids():
    assert _pins_from_notebook() == EXPECTED_PINS


def test_notebook_still_enforces_reviewed_and_human():
    """The filters are correct and must stay on; ambiguity is not caused by them."""
    src = _NB.read_text()
    assert 'SP = "UniProtKB-Swiss-Prot"' in src, "reviewed-only server filter removed"
    assert 'idmap(entrez, "GeneID", to_db=SP)' in src, "Entrez route lost its reviewed filter"
    assert "_pick(_entrez_raw, human_only=True)" in src, "Entrez route lost its human filter"


def test_notebook_raises_on_unpinned_divergent_ambiguity():
    src = _NB.read_text()
    assert "_unpinned_divergent" in src and "raise ValueError" in src, \
        "the mapping must hard-error on a new unpinned divergent GeneID"


def _classify(amb, pins):
    """Reimplementation of the notebook's rule, for behavioural testing."""
    n = amb.groupby("From")["Entry"].nunique()
    n = n[n > 1]
    identical, divergent = [], []
    for e in sorted(set(n.index)):
        seqs = set(amb.loc[amb["From"] == e, "Sequence"].fillna(""))
        (identical if len(seqs) == 1 else divergent).append(e)
    return identical, divergent, [e for e in divergent if e not in pins]


@pytest.fixture
def real_amb():
    if not _AMB.exists():
        pytest.skip(f"{_AMB} not present (requires the mapping outputs)")
    d = pd.read_csv(_AMB)
    d["From"] = d["From"].astype(str)
    return d


def test_real_data_has_no_unpinned_divergent_geneid(real_amb):
    identical, divergent, unpinned = _classify(real_amb, EXPECTED_PINS)
    assert sorted(divergent) == sorted(EXPECTED_PINS), \
        f"the set of divergent GeneIDs changed: {divergent}"
    assert unpinned == [], f"unpinned divergent GeneIDs: {unpinned}"
    # PRR20 (221 aa x5) and CT45A (189 aa x3): identical sequences, pick immaterial
    assert identical, "expected the sequence-identical paralog families"


def _row(frm, entry, gene, seq):
    return {"From": frm, "Entry": entry, "Reviewed": "reviewed",
            "Organism (ID)": 9606, "Gene Names (primary)": gene, "Sequence": seq}


def test_new_divergent_geneid_is_caught(real_amb):
    extra = pd.DataFrame([_row("999999", "P00001", "FAKEA", "M" * 100),
                          _row("999999", "P00002", "FAKEB", "M" * 250)])
    _, _, unpinned = _classify(pd.concat([real_amb, extra]), EXPECTED_PINS)
    assert unpinned == ["999999"]


def test_new_sequence_identical_family_passes(real_amb):
    """A paralog family is ambiguous but harmless -- it must not raise."""
    extra = pd.DataFrame([_row("888888", "P00003", "FAMA", "M" * 150),
                          _row("888888", "P00004", "FAMB", "M" * 150)])
    identical, _, unpinned = _classify(pd.concat([real_amb, extra]), EXPECTED_PINS)
    assert unpinned == []
    assert "888888" in identical
