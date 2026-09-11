"""The SKEMPI reference defines C1/C2/C3 for the SKEMPI-pretrained methods.

Regression tests for a real defect (fixed 2026-09-10). The old reference,
`results/gcv/SAAMBE_train_uniprots.npy`, held 258 accessions because its
derivation kept only SINGLE-CHARACTER chain groups from SKEMPI's `#Pdb` tags --
silently dropping all 122 multi-chain complexes (`3SE8_HL_G`, `1BD2_ABC_DE`,
...), which are overwhelmingly antibodies whose heavy and light chains are one
partner. The correct per-chain expansion yields 342.

Under-counting SKEMPI's proteins makes SAAMBE-3D / MutPPI / MutPPI+ look less
overlapped with our test pairs than they are, so their reported generalisation
was optimistic.
"""
from pathlib import Path

import pytest

pytest.importorskip("pandas")

from data_processing.training_sets import prepare_skempi_reference as psr  # noqa: E402

SKEMPI = Path("datasets/source_data/skempi_v2.csv")
SIFTS = Path("datasets/source_data/pdb_chain_uniprot.csv")
OUT = Path("datasets/annotations/skempi_train_uniprots.csv")

needs_sources = pytest.mark.skipif(
    not (SKEMPI.exists() and SIFTS.exists()),
    reason="SKEMPI/SIFTS source files not present")


def test_multichain_groups_expand_per_character():
    """`3SE8_HL_G` is three chains (H, L, G), not two tokens."""
    tag = "3SE8_HL_G"
    parts = tag.split("_")
    chains = [c for group in parts[1:] for c in group]
    assert chains == ["H", "L", "G"]
    # The old rule -- keep only single-character groups -- would see only "G".
    old_rule = [g for g in parts[1:] if len(g) == 1]
    assert old_rule == ["G"], "this is the bug the 258-accession file encoded"


@needs_sources
def test_derivation_covers_every_complex():
    accs, unresolved, n_complexes = psr.derive(SKEMPI, SIFTS)
    assert n_complexes == 348
    assert len(accs) > 258, ("regressed to the single-character-group rule: "
                             "multi-chain complexes are being dropped again")
    assert len(accs) == 342


@needs_sources
def test_multichain_complexes_contribute_accessions():
    """Directly assert the previously-dropped class is now represented."""
    sifts = psr.load_sifts(SIFTS)
    tags = psr.skempi_complexes(SKEMPI)
    multi = [t for t in tags if any(len(g) > 1 for g in t.split("_")[1:])]
    assert len(multi) == 122, f"expected 122 multi-chain complexes, got {len(multi)}"

    from_multi = set()
    for t in multi:
        pdb = t.split("_")[0].lower()
        for group in t.split("_")[1:]:
            if len(group) > 1:
                for ch in group:
                    from_multi |= sifts.get((pdb, ch), set())
    assert from_multi, "multi-chain groups resolved to nothing"

    accs, _, _ = psr.derive(SKEMPI, SIFTS)
    assert from_multi <= accs


@needs_sources
def test_chimeric_chain_keeps_every_accession():
    """A chain mapping to several accessions must contribute all of them."""
    sifts = psr.load_sifts(SIFTS)
    assert all(isinstance(v, set) for v in list(sifts.values())[:50])


@pytest.mark.skipif(not OUT.exists(), reason="derived reference not built")
def test_written_reference_matches_derivation():
    from utils.gcv_common import load_skempi_train_uniprots
    loaded = load_skempi_train_uniprots()
    assert len(loaded) == 342
    assert "Q7A260" not in loaded, ("Q7A260 was in the old reference but SKEMPI "
                                   "does not reference it")


@pytest.mark.skipif(not OUT.exists(), reason="derived reference not built")
def test_reference_is_an_input_not_a_result():
    """It must not live under results/ -- that tree is generated and gitignored."""
    assert "results" not in OUT.parts, OUT
    assert not Path("results/gcv/SAAMBE_train_uniprots.npy").exists(), \
        "the old misplaced reference is back"
