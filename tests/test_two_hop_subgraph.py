"""Tests for the two-hop restriction used by MutPred-PPI training and inference.

`model.GAT_mut_processor` reads a single node -- `h[mutation_idx]` after two GAT
layers -- and pools nothing. Its output is therefore a pure function of the
mutation site's two-hop neighbourhood, so restricting each sample to that
neighbourhood is exact rather than approximate. `compress_to_subgraphs.py`
already relies on this for inference; `mutpred_ppi_data.build_tensors` now does
the same for training.

These tests pin the properties the argument depends on:

  * the kept set is exactly the two-hop closed neighbourhood,
  * the subgraph is INDUCED, so every retained node's neighbour set is complete
    and no attention softmax is taken over a truncated neighbourhood,
  * indices are remapped consistently, and
  * the model's output is unchanged -- the property that makes the optimisation
    legitimate rather than merely fast.
"""
import numpy as np
import pytest
import torch

from contact_graphs import two_hop_subgraph
from model import GAT_mut_processor


def _undirected(edges, n_nodes):
    """(2, E) edge_index with both directions and self-loops, as the store emits."""
    pairs = set()
    for a, b in edges:
        pairs.add((a, b))
        pairs.add((b, a))
    for i in range(n_nodes):
        pairs.add((i, i))
    arr = np.array(sorted(pairs), dtype=np.int64).T
    return arr


# 0-1-2-3-4 path, plus an isolated-ish 5 attached to 4.
_PATH = _undirected([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 6)


class TestNeighbourhood:
    def test_keeps_exactly_the_two_hop_closed_neighbourhood(self):
        keep, _, _ = two_hop_subgraph(_PATH, 0)
        assert list(keep) == [0, 1, 2]

    def test_centre_in_the_middle_reaches_both_directions(self):
        keep, _, _ = two_hop_subgraph(_PATH, 2)
        assert list(keep) == [0, 1, 2, 3, 4]

    def test_three_hop_nodes_are_excluded(self):
        keep, _, _ = two_hop_subgraph(_PATH, 0)
        assert 3 not in keep and 4 not in keep and 5 not in keep

    def test_centre_is_always_present(self):
        for c in range(6):
            keep, _, local = two_hop_subgraph(_PATH, c)
            assert c in keep
            assert keep[local] == c


class TestInducedAndRemapped:
    def test_subgraph_is_induced(self):
        """Every original edge between two kept nodes survives.

        This is what keeps each retained node's neighbour set complete; dropping
        any such edge would renormalise an attention softmax over a truncated
        neighbourhood and silently change the result.
        """
        keep, local, _ = two_hop_subgraph(_PATH, 2)
        keep_set = set(keep.tolist())
        expected = sum(1 for a, b in zip(_PATH[0], _PATH[1])
                       if a in keep_set and b in keep_set)
        assert local.shape[1] == expected

    def test_local_indices_are_dense_and_in_range(self):
        keep, local, local_centre = two_hop_subgraph(_PATH, 2)
        assert local.min() >= 0
        assert local.max() == len(keep) - 1
        assert 0 <= local_centre < len(keep)

    def test_local_edges_match_original_edges_under_the_mapping(self):
        keep, local, _ = two_hop_subgraph(_PATH, 2)
        back = {(int(keep[a]), int(keep[b])) for a, b in zip(local[0], local[1])}
        orig = {(int(a), int(b)) for a, b in zip(_PATH[0], _PATH[1])
                if a in set(keep.tolist()) and b in set(keep.tolist())}
        assert back == orig

    def test_self_loops_survive(self):
        keep, local, centre = two_hop_subgraph(_PATH, 2)
        assert any(a == centre and b == centre for a, b in zip(local[0], local[1]))


class TestModelOutputUnchanged:
    """The property the optimisation actually rests on."""

    @staticmethod
    def _run(model, x, edge_index, mut_idx, mut_diff):
        with torch.no_grad():
            return model(torch.tensor(x, dtype=torch.float),
                         torch.tensor(edge_index, dtype=torch.long),
                         mut_idx,
                         torch.tensor(mut_diff, dtype=torch.float)).item()

    @pytest.mark.parametrize("centre", [0, 2, 4])
    def test_full_graph_and_subgraph_agree(self, centre):
        torch.manual_seed(0)
        rng = np.random.default_rng(centre)
        model = GAT_mut_processor(32).eval()
        n = _PATH.max() + 1
        x = rng.standard_normal((n, 32)).astype(np.float32)
        mut_diff = rng.standard_normal(1024).astype(np.float32)

        full = self._run(model, x, _PATH, centre, mut_diff)
        keep, local, local_centre = two_hop_subgraph(_PATH, centre)
        sub = self._run(model, x[keep], local, local_centre, mut_diff)
        assert full == pytest.approx(sub, abs=1e-5)

    def test_agreement_holds_on_a_denser_random_graph(self):
        torch.manual_seed(1)
        rng = np.random.default_rng(7)
        n = 60
        edges = {(int(a), int(b)) for a, b in
                 rng.integers(0, n, size=(300, 2)) if a != b}
        ei = _undirected(sorted(edges), n)
        model = GAT_mut_processor(32).eval()
        x = rng.standard_normal((n, 32)).astype(np.float32)
        mut_diff = rng.standard_normal(1024).astype(np.float32)

        for centre in (0, 13, 41):
            full = self._run(model, x, ei, centre, mut_diff)
            keep, local, lc = two_hop_subgraph(ei, centre)
            sub = self._run(model, x[keep], local, lc, mut_diff)
            assert full == pytest.approx(sub, abs=1e-5), f"centre {centre}"

    def test_one_hop_is_insufficient(self):
        """Negative control for the two tests above.

        Restricting to ONE hop changes the answer. Without this, a model that
        ignored its graph entirely would satisfy "full and subgraph agree" while
        proving nothing. Together the three tests say the model genuinely uses
        its graph AND two hops is the exact depth its output depends on.
        """
        torch.manual_seed(2)
        rng = np.random.default_rng(3)
        model = GAT_mut_processor(32).eval()
        n = _PATH.max() + 1
        x = rng.standard_normal((n, 32)).astype(np.float32)
        mut_diff = rng.standard_normal(1024).astype(np.float32)

        centre = 2
        full = self._run(model, x, _PATH, centre, mut_diff)
        one_hop = np.array([1, 2, 3])
        mask = np.isin(_PATH[0], one_hop) & np.isin(_PATH[1], one_hop)
        remap = np.full(n, -1, dtype=np.int64)
        remap[one_hop] = np.arange(len(one_hop))
        local = np.stack([remap[_PATH[0][mask]], remap[_PATH[1][mask]]])
        truncated = self._run(model, x[one_hop], local, int(remap[centre]), mut_diff)
        assert full != pytest.approx(truncated, abs=1e-5)
