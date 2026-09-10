"""Step 00 and step 01 of the public pipeline must agree on the separator.

Regression tests for a real break (fixed 2026-09-10): `00_make_af3_json_input.py`
names every AlphaFold3 job `{id_a}__{id_b}`, but `find_mmcif_file` in
`01_make_contact_graphs_and_fasta.py` tried only `{a}_{b}` and `{a}-{b}` joins
and had no `__` case at all. A user who followed `docs/INFERENCE.md` step 1,
folded the JSONs, and ran step 2 hit `FileNotFoundError` on every pair.

The `__` join exists because a single `-` is ambiguous for isoform accessions:
`O14787-2-Q13207` cannot be split back into two accessions. So the isoform case
below is the one that actually motivates the whole convention.
"""
import glob
import os
import re

import pytest

_SRC = "src/inference/01_make_contact_graphs_and_fasta.py"


@pytest.fixture(scope="module")
def find_mmcif_file():
    """Extract the resolver without importing the module (it is a CLI script)."""
    src = open(_SRC).read()
    start = src.index("# AF3 wraps the JSON `name` field")
    marker = "    return list(unique_files.values())[0]"
    end = src.index(marker) + len(marker) + 1
    ns = {"os": os, "glob": glob, "re": re}
    exec(compile(src[start:end], _SRC, "exec"), ns)
    return ns["find_mmcif_file"]


@pytest.fixture
def mmcif_dir(tmp_path):
    for name in ("P12345__Q67890.cif",                     # step 00's own output
                 "fold_o14787-2__q13207_model_0.cif",      # AF3-wrapped isoform
                 "A11111_B22222.cif",                      # legacy single _
                 "fold_c33333-d44444_model_0.cif"):        # legacy single -
        (tmp_path / name).write_text("")
    return str(tmp_path)


@pytest.mark.parametrize("a,b,expected,swapped", [
    ("P12345", "Q67890", "P12345__Q67890.cif", False),
    ("Q67890", "P12345", "P12345__Q67890.cif", True),
    ("A11111", "B22222", "A11111_B22222.cif", False),
    ("B22222", "A11111", "A11111_B22222.cif", True),
    ("C33333", "D44444", "fold_c33333-d44444_model_0.cif", False),
])
def test_resolves_pair(find_mmcif_file, mmcif_dir, a, b, expected, swapped):
    path, sw = find_mmcif_file(a, b, mmcif_dir)
    assert os.path.basename(path) == expected
    assert sw is swapped


@pytest.mark.parametrize("a,b,swapped", [
    ("O14787-2", "Q13207", False),
    ("Q13207", "O14787-2", True),
])
def test_isoform_accession_resolves_unambiguously(find_mmcif_file, mmcif_dir,
                                                  a, b, swapped):
    """The case the `__` convention exists for; unresolvable before the fix."""
    path, sw = find_mmcif_file(a, b, mmcif_dir)
    assert os.path.basename(path) == "fold_o14787-2__q13207_model_0.cif"
    assert sw is swapped


def test_missing_pair_error_names_the_expected_filename(find_mmcif_file, mmcif_dir):
    with pytest.raises(FileNotFoundError, match=r"P99999__Q88888"):
        find_mmcif_file("P99999", "Q88888", mmcif_dir)


def test_isoform_pair_does_not_match_a_different_isoform(find_mmcif_file, tmp_path):
    """`O14787-2__Q13207` must not satisfy a request for `O14787__Q13207`.

    The bare accession and its isoform are different proteins with different
    sequences; a substring match would happily conflate them.
    """
    (tmp_path / "O14787-2__Q13207.cif").write_text("")
    with pytest.raises(FileNotFoundError):
        find_mmcif_file("O14787", "Q13207", str(tmp_path))


def test_step00_and_step01_agree_on_the_separator():
    """Pin the contract itself, so the two halves cannot drift apart again."""
    step00 = open("src/inference/00_make_af3_json_input.py").read()
    assert 'f"{id_a}__{id_b}"' in step00, "step 00 changed its pair-naming scheme"
    assert '"__" in stem' in open(_SRC).read(), "step 01 lost its `__` case"
