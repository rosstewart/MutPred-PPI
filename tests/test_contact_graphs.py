"""Tests for contact_graphs.py -- the MAP residue policy and the one contact
graph builder.

The synthetic PDB in `_SYNTH_PDB` is deliberately hand-written rather than a
real structure: no real input in this repo exercises a non-standard residue
(measured 0 across all 862 stability PDBs and a 200-file AF3 sample), so
without a synthetic case the MAP policy -- and specifically the `UNK` trap --
would be untested by construction.
"""
import numpy as np
import pytest

from contact_graphs import (
    _is_polymer_residue,
    contact_graph_from_structure,
    dense_from_edges,
    edges_from_dense,
    pair_key,
    residue_to_one,
    sha,
    symmetric_edge_index,
)

# Chain A: ALA(A, standard) - MSE(M, modified -> maps to parent) -
#          UNK(X, unidentified -> must hold its position) - HOH (water, excluded)
# Chain B: GLY(G, standard)
# Coordinates are close enough that A-B and within-chain neighbours make
# contacts at the default 4.5 A threshold; exact edges are not asserted here,
# only that residues map/exclude correctly and both chains are found.
_SYNTH_PDB = """\
ATOM      1  N   ALA A   1      11.104  13.207   2.158  1.00 20.00           N
ATOM      2  CA  ALA A   1      11.914  13.334   0.938  1.00 20.00           C
ATOM      3  C   ALA A   1      13.404  13.100   1.218  1.00 20.00           C
ATOM      4  O   ALA A   1      13.870  13.402   2.318  1.00 20.00           O
ATOM      5  CB  ALA A   1      11.428  12.319  -0.100  1.00 20.00           C
HETATM    6  N   MSE A   2      14.104  12.540   0.250  1.00 20.00           N
HETATM    7  CA  MSE A   2      15.550  12.310   0.340  1.00 20.00           C
HETATM    8  C   MSE A   2      16.230  13.610   0.780  1.00 20.00           C
HETATM    9  O   MSE A   2      15.700  14.700   0.900  1.00 20.00           O
HETATM   10 SE   MSE A   2      16.100  11.100  -1.100  1.00 20.00          SE
ATOM     11  N   UNK A   3      17.700  13.500   1.000  1.00 20.00           N
ATOM     12  CA  UNK A   3      18.500  14.700   1.200  1.00 20.00           C
ATOM     13  C   UNK A   3      19.900  14.400   1.700  1.00 20.00           C
ATOM     14  O   UNK A   3      20.300  13.300   2.000  1.00 20.00           O
HETATM   15  O   HOH A   4      25.000  25.000  25.000  1.00 20.00           O
ATOM     16  N   GLY B   1      30.000  30.000  30.000  1.00 20.00           N
ATOM     17  CA  GLY B   1      31.000  30.500  30.500  1.00 20.00           C
ATOM     18  C   GLY B   1      32.000  30.000  31.000  1.00 20.00           C
ATOM     19  O   GLY B   1      33.000  30.500  31.500  1.00 20.00           O
TER
END
"""


@pytest.fixture
def synth_pdb_path(tmp_path):
    path = tmp_path / "synth.pdb"
    path.write_text(_SYNTH_PDB)
    return path


# -- residue_to_one: the MAP policy itself ---------------------------------------

def test_residue_to_one_standard():
    assert residue_to_one("ALA") == "A"
    assert residue_to_one("GLY") == "G"


def test_residue_to_one_modified_maps_to_parent():
    assert residue_to_one("MSE") == "M"  # selenomethionine -> methionine


def test_residue_to_one_unknown_maps_to_x():
    assert residue_to_one("UNK") == "X"
    assert residue_to_one("SOME_MADE_UP_CODE") == "X"


# -- _is_polymer_residue: what counts as a sequence position ---------------------

class _FakeResidue:
    def __init__(self, resname):
        self._resname = resname

    def get_resname(self):
        return self._resname


def test_is_polymer_residue_standard_and_modified():
    assert _is_polymer_residue(_FakeResidue("ALA"))
    assert _is_polymer_residue(_FakeResidue("MSE"))


def test_is_polymer_residue_unk_counts_the_unk_trap():
    # UNK is NOT is_aa() under either the default or standard=True test, so a
    # naive `if not is_aa(residue): continue` would drop it -- reintroducing
    # the exact frame shift the MAP policy exists to close.
    assert _is_polymer_residue(_FakeResidue("UNK"))


def test_is_polymer_residue_excludes_water():
    assert not _is_polymer_residue(_FakeResidue("HOH"))


# -- contact_graph_from_structure: MAP policy end-to-end -------------------------

def test_contact_graph_default_maps_and_preserves_length(synth_pdb_path):
    result = contact_graph_from_structure(synth_pdb_path)
    assert result is not None
    seq_a, seq_b, edge_index = result
    # ALA-MSE-UNK-HOH: HOH excluded (not polymer), the other three held with
    # MSE mapped to M and UNK mapped to X -- length is exactly 3, not 2 or 4.
    assert seq_a == "AMX"
    assert seq_b == "G"


def test_contact_graph_monomer_mode(synth_pdb_path):
    result = contact_graph_from_structure(synth_pdb_path, chains=("A",))
    assert result is not None
    seq_a, seq_b, _ = result
    assert seq_a == "AMX"
    assert seq_b == ""


def test_contact_graph_explicit_two_chains(synth_pdb_path):
    result = contact_graph_from_structure(synth_pdb_path, chains=("A", "B"))
    assert result is not None
    seq_a, seq_b, _ = result
    assert seq_a == "AMX"
    assert seq_b == "G"


def test_contact_graph_missing_requested_chain_returns_none(synth_pdb_path):
    assert contact_graph_from_structure(synth_pdb_path, chains=("A", "Z")) is None


def test_contact_graph_missing_file_returns_none(tmp_path):
    assert contact_graph_from_structure(tmp_path / "does_not_exist.pdb") is None


def test_contact_graph_edge_index_shape(synth_pdb_path):
    _, _, edge_index = contact_graph_from_structure(synth_pdb_path)
    assert edge_index.shape[0] == 2
    n = len("AMX") + len("G")
    if edge_index.size:
        assert edge_index.max() < n
        assert edge_index.min() >= 0
        assert np.all(edge_index[0] < edge_index[1])  # upper triangle only


# -- pair_key --------------------------------------------------------------------

def test_pair_key_order_independent():
    assert pair_key("SEQA", "SEQB") == pair_key("SEQB", "SEQA")


def test_pair_key_monomer_has_distinct_prefix():
    key = pair_key("SEQA", "")
    assert key.startswith("mono_")
    assert key != pair_key("SEQA", "SEQA")  # a homodimer is not a monomer


def test_pair_key_matches_sha():
    ha, hb = sha("SEQA"), sha("SEQB")
    assert pair_key("SEQA", "SEQB") == f"{min(ha, hb)}_{max(ha, hb)}"


# -- edge helpers ------------------------------------------------------------------

def test_edges_from_dense_upper_triangle_only():
    G = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    edges = edges_from_dense(G)
    assert edges.shape == (2, 2)
    assert np.all(edges[0] < edges[1])


def test_dense_from_edges_round_trip():
    G = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    edges = edges_from_dense(G)
    back = dense_from_edges(edges, n=3)
    assert np.array_equal(back, G)


def test_symmetric_edge_index_adds_self_loops_and_both_directions():
    edges = np.array([[0], [1]])  # single edge 0-1
    sym = symmetric_edge_index(edges, n=3)
    pairs = set(zip(sym[0].tolist(), sym[1].tolist()))
    assert (0, 1) in pairs and (1, 0) in pairs
    assert (0, 0) in pairs and (1, 1) in pairs and (2, 2) in pairs


def test_symmetric_edge_index_idempotent_on_stored_diagonal():
    # A record that already carries a diagonal reads back identical to one
    # that does not.
    no_diag = np.array([[0], [1]])
    with_diag = np.array([[0, 0], [1, 0]])
    assert np.array_equal(
        symmetric_edge_index(no_diag, n=2), symmetric_edge_index(with_diag, n=2))
