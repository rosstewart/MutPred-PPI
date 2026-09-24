"""The per-residue RSA/pLDDT cache must agree with the structures it came from.

`build_residue_cache.py` is the only source of solvent accessibility in the
repository, and Supplementary Fig S5A's surface-vs-buried split rests entirely
on it. Nothing else cross-checks it, so these assert the properties that would
silently move the figure if they broke:

  * RSA is a ratio against the Tien et al. 2013 maxima, so it must be positive
    and the buried fraction must land in the range a folded protein gives.
  * pLDDT is a percentage read from the mmCIF B-factor column. The historical
    failure here is a fixed column offset returning -1.0 (see
    build_plddt_cache.py), which a range check catches.
  * The per-complex mean pLDDT must track the manifest's. It cannot match
    EXACTLY: the manifest is atom-weighted and a per-residue cache is
    residue-weighted, so the test bounds the gap rather than requiring equality.
  * dSASA must be non-negative -- binding cannot expose surface -- and interface
    residues must be less buried than the rest.
"""
import pickle

import numpy as np
import pandas as pd
import pytest

from paths import ANNOTATIONS_DIR, DATASETS_DIR

CACHE = ANNOTATIONS_DIR / "residue_cache.pkl"
MANIFEST = DATASETS_DIR / "af3_structures_canonical" / "manifest.csv"

#: The manifest's `mean_plddt` is atom-weighted, this cache is residue-weighted.
#: Observed over all 2,604 complexes: mean 0.45, max 3.79.
MAX_PLDDT_WEIGHTING_GAP = 6.0

needs_cache = pytest.mark.skipif(not CACHE.exists(),
                                 reason="residue_cache.pkl not built")


@pytest.fixture(scope="module")
def cache():
    with open(CACHE, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def flat(cache):
    def stack(field):
        return np.concatenate([c[field] for rec in cache.values()
                               for c in rec.values()])
    return {f: stack(f) for f in ("rsa", "dsasa", "plddt")}


@needs_cache
def test_rsa_is_a_plausible_ratio(flat):
    rsa = flat["rsa"][np.isfinite(flat["rsa"])]
    assert (rsa >= 0).all(), "negative RSA: SASA cannot be below zero"
    buried = float((rsa < 0.25).mean())
    assert 0.15 < buried < 0.40, (
        f"{buried:.1%} of residues are buried at RSA < 0.25; a folded protein "
        f"gives roughly a quarter, so the max-ASA reference is likely wrong")


@needs_cache
def test_plddt_is_a_percentage(flat):
    plddt = flat["plddt"][np.isfinite(flat["plddt"])]
    assert plddt.min() >= 0.0 and plddt.max() <= 100.0, (
        f"pLDDT outside [0, 100]: {plddt.min():.2f}..{plddt.max():.2f}. The -1.0 "
        f"sentinel means the mmCIF B-factor column was located by offset rather "
        f"than by name.")


@needs_cache
def test_binding_never_exposes_surface(flat):
    dsasa = flat["dsasa"][np.isfinite(flat["dsasa"])]
    # A small negative tail is numerical noise in the Shrake-Rupley sphere
    # sampling, not a residue that gained surface on binding.
    assert dsasa.min() > -1.0, f"dSASA as low as {dsasa.min():.2f} A^2"


@needs_cache
def test_interface_residues_are_less_buried_than_the_rest(flat):
    iface = flat["dsasa"] > 1.0
    buried_iface = float((flat["rsa"][iface] < 0.25).mean())
    buried_rest = float((flat["rsa"][~iface] < 0.25).mean())
    assert buried_iface < buried_rest, (
        f"residues that lose surface on binding are {buried_iface:.1%} buried "
        f"against {buried_rest:.1%} for the rest; the two SASA passes may have "
        f"been swapped")


@needs_cache
@pytest.mark.skipif(not MANIFEST.exists(), reason="structure manifest not present")
def test_per_complex_mean_plddt_tracks_the_manifest(cache):
    man = pd.read_csv(MANIFEST, usecols=["chain_a_accession", "chain_b_accession",
                                         "mean_plddt", "len_a", "len_b"])
    man["key"] = man.chain_a_accession + "__" + man.chain_b_accession
    man = man.drop_duplicates("key").set_index("key")

    gaps, bad_lengths = [], []
    for key, record in cache.items():
        if key not in man.index:
            continue
        row = man.loc[key]
        mine = float(np.mean(np.concatenate([c["plddt"] for c in record.values()])))
        gaps.append(abs(mine - float(row.mean_plddt)))
        want = {row.chain_a_accession: int(row.len_a),
                row.chain_b_accession: int(row.len_b)}
        got = {acc: len(c["plddt"]) for acc, c in record.items()}
        if got != want:
            bad_lengths.append((key, got, want))

    assert gaps, "no cached complex is in the manifest"
    assert not bad_lengths, (
        f"{len(bad_lengths)} complexes have a chain length disagreeing with the "
        f"manifest, e.g. {bad_lengths[0]}")
    assert max(gaps) < MAX_PLDDT_WEIGHTING_GAP, (
        f"per-complex mean pLDDT differs from the manifest by up to "
        f"{max(gaps):.2f}, beyond the atom- vs residue-weighting gap")
