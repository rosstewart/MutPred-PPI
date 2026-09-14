"""Does block-diagonal batching reproduce the per-sample forward exactly?

MutPred-PPI trains one sample per step: roughly 1,400 samples x 100 epochs is
~140,000 separate forward passes per fold, and at ~30-node two-hop subgraphs the
launch overhead dominates the arithmetic. Batching would cut that by the batch
factor.

Graphs of unequal size batch by CONCATENATION, not padding: several graphs become
one disconnected graph and the edge indices are offset. Message passing never
crosses components, so every node sees exactly the neighbourhood it would have
seen alone.

Nothing in `src/` is modified by this file. `_batched_forward` below reimplements
the forward over the model's existing submodules, so these tests act as the
specification for the change to `model.GAT_mut_processor.forward` -- which today
reads a single site (`h[mutation_idx:mutation_idx + 1]`) and would need to accept
several. If this file is green, that change is safe to make.
"""
import numpy as np
import pytest
import torch

from model import GAT_mut_processor

D = 32


def _random_graph(rng, n):
    """(x, edge_index, centre) with both edge directions and self-loops."""
    e = {(i, i) for i in range(n)}
    for a, b in rng.integers(0, n, size=(n * 3, 2)):
        if a != b:
            e.add((int(a), int(b)))
            e.add((int(b), int(a)))
    ei = np.array(sorted(e), dtype=np.int64).T
    x = rng.standard_normal((n, D)).astype(np.float32)
    return x, ei, int(rng.integers(0, n))


def _collate(graphs):
    """Concatenate graphs into one disconnected graph, offsetting indices."""
    xs, eis, centres, offset = [], [], [], 0
    for x, ei, k in graphs:
        xs.append(x)
        eis.append(ei + offset)
        centres.append(k + offset)
        offset += x.shape[0]
    return (torch.tensor(np.concatenate(xs)),
            torch.tensor(np.concatenate(eis, axis=1)),
            centres)


def _batched_forward(model, X, EI, centres, mut_diff):
    """What `forward` would do given several mutation sites at once.

    Identical to the released forward except that the site is a list rather than
    a scalar: two GAT layers over everything, then read each centre.
    """
    h = torch.relu(model.complex_gat1(X, EI))
    h = torch.relu(model.complex_gat2(h, EI))
    pm = model.mutation_diff_processor(mut_diff.unsqueeze(0))
    return [model.binding_predictor(
        torch.cat([h[k:k + 1], pm], dim=-1)).squeeze() for k in centres]


def _single_forward(model, graphs, mut_diff):
    out = []
    for x, ei, k in graphs:
        out.append(model(torch.tensor(x), torch.tensor(ei), k, mut_diff).squeeze())
    return out


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GAT_mut_processor(D).eval()


class TestEquivalence:
    @pytest.mark.parametrize("sizes", [[7, 23, 12], [5, 5, 5], [1, 40], [64, 3, 18, 9]])
    def test_unequal_sizes_batch_exactly(self, model, sizes):
        """The case that looks like it should fail: graphs of different sizes."""
        rng = np.random.default_rng(len(sizes))
        graphs = [_random_graph(rng, n) for n in sizes]
        md = torch.tensor(rng.standard_normal(1024).astype(np.float32))
        with torch.no_grad():
            single = _single_forward(model, graphs, md)
            X, EI, centres = _collate(graphs)
            batched = _batched_forward(model, X, EI, centres, md)
        for a, b in zip(single, batched):
            assert a.item() == pytest.approx(b.item(), abs=1e-6)

    def test_batch_of_one_matches_the_scalar_path(self, model):
        """The released single-site forward is the batch-of-one special case."""
        rng = np.random.default_rng(99)
        g = _random_graph(rng, 11)
        md = torch.tensor(rng.standard_normal(1024).astype(np.float32))
        with torch.no_grad():
            single = model(torch.tensor(g[0]), torch.tensor(g[1]), g[2], md).squeeze()
            X, EI, centres = _collate([g])
            batched = _batched_forward(model, X, EI, centres, md)[0]
        assert single.item() == pytest.approx(batched.item(), abs=1e-7)

    def test_order_within_the_batch_does_not_matter(self, model):
        rng = np.random.default_rng(5)
        graphs = [_random_graph(rng, n) for n in (9, 17, 6)]
        md = torch.tensor(rng.standard_normal(1024).astype(np.float32))
        with torch.no_grad():
            X, EI, c = _collate(graphs)
            fwd = _batched_forward(model, X, EI, c, md)
            Xr, EIr, cr = _collate(graphs[::-1])
            rev = _batched_forward(model, Xr, EIr, cr, md)
        for a, b in zip(fwd, rev[::-1]):
            assert a.item() == pytest.approx(b.item(), abs=1e-6)


class TestGradientsMatch:
    """Training is the use case, so the backward pass has to agree too."""

    def test_gradients_equal_the_sum_of_per_sample_gradients(self, model):
        rng = np.random.default_rng(11)
        graphs = [_random_graph(rng, n) for n in (8, 19, 13)]
        md = torch.tensor(rng.standard_normal(1024).astype(np.float32))
        target = torch.tensor([1.0, 0.0, 1.0])
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

        model.zero_grad()
        outs = _single_forward(model, graphs, md)
        loss_fn(torch.stack(outs), target).backward()
        per_sample = {n: p.grad.detach().clone()
                      for n, p in model.named_parameters() if p.grad is not None}

        model.zero_grad()
        X, EI, centres = _collate(graphs)
        outs_b = _batched_forward(model, X, EI, centres, md)
        loss_fn(torch.stack(outs_b), target).backward()

        assert per_sample, "no gradients captured"
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            torch.testing.assert_close(p.grad, per_sample[n], rtol=1e-4, atol=1e-6,
                                       msg=f"gradient mismatch for {n}")


class TestNegativeControls:
    """Guard the agreement tests against passing for the wrong reason."""

    def test_forgetting_to_offset_edges_changes_the_answer(self, model):
        """The mistake batching invites: edges pointing into the wrong graph."""
        rng = np.random.default_rng(21)
        graphs = [_random_graph(rng, n) for n in (10, 14)]
        md = torch.tensor(rng.standard_normal(1024).astype(np.float32))
        with torch.no_grad():
            X, EI, centres = _collate(graphs)
            correct = _batched_forward(model, X, EI, centres, md)
            # concatenate WITHOUT offsetting the second graph's edges
            bad_ei = torch.tensor(np.concatenate([graphs[0][1], graphs[1][1]], axis=1))
            wrong = _batched_forward(model, X, bad_ei, centres, md)
        assert any(a.item() != pytest.approx(b.item(), abs=1e-6)
                   for a, b in zip(correct, wrong))

    def test_connecting_the_components_changes_the_answer(self, model):
        """Batching is only exact because the components stay disconnected."""
        rng = np.random.default_rng(31)
        graphs = [_random_graph(rng, n) for n in (10, 14)]
        md = torch.tensor(rng.standard_normal(1024).astype(np.float32))
        X, EI, centres = _collate(graphs)
        bridge = torch.tensor([[centres[0], centres[1]],
                               [centres[1], centres[0]]], dtype=torch.long)
        with torch.no_grad():
            correct = _batched_forward(model, X, EI, centres, md)
            joined = _batched_forward(model, X, torch.cat([EI, bridge], dim=1),
                                      centres, md)
        assert any(a.item() != pytest.approx(b.item(), abs=1e-6)
                   for a, b in zip(correct, joined))


class TestTrainingMode:
    """`train_fold` already batches the OPTIMISER; only the forwards are serial.

    The inner loop buffers `batch_size` logits and takes ONE step on the stacked
    loss (`train_fold.py:282-291`). So fusing the forwards does not change the
    optimisation -- same loss, same gradient, same single step. What it does
    change is dropout: 16 separate forwards draw 16 masks from the generator in
    sequence, while one batched forward draws a single (16, D) mask. Same
    distribution, different draw. These two tests pin down that boundary so the
    speedup is not mistaken for bit-identity under dropout.
    """

    @staticmethod
    def _no_dropout(model):
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
        return model

    def test_exact_in_train_mode_once_dropout_is_disabled(self, model):
        """With p=0, train mode and eval mode agree, so batching stays exact."""
        rng = np.random.default_rng(41)
        graphs = [_random_graph(rng, n) for n in (12, 25, 8)]
        md = torch.tensor(rng.standard_normal(1024).astype(np.float32))
        self._no_dropout(model).train()
        with torch.no_grad():
            single = _single_forward(model, graphs, md)
            X, EI, centres = _collate(graphs)
            batched = _batched_forward(model, X, EI, centres, md)
        for a, b in zip(single, batched):
            assert a.item() == pytest.approx(b.item(), abs=1e-6)

    def test_active_dropout_makes_the_two_paths_differ(self, model):
        """Documents the one real caveat: under dropout this is not bit-identical.

        Not a defect -- the masks are drawn from the same distribution -- but it
        means a batched training run reproduces the per-sample run only in
        distribution, which is already true of any two GPU runs of this model.
        """
        rng = np.random.default_rng(43)
        graphs = [_random_graph(rng, n) for n in (12, 25, 8)]
        md = torch.tensor(rng.standard_normal(1024).astype(np.float32))
        model.train()
        with torch.no_grad():
            torch.manual_seed(7)
            single = _single_forward(model, graphs, md)
            torch.manual_seed(7)
            X, EI, centres = _collate(graphs)
            batched = _batched_forward(model, X, EI, centres, md)
        assert any(a.item() != pytest.approx(b.item(), abs=1e-6)
                   for a, b in zip(single, batched))

    def test_the_stacked_loss_and_its_gradient_are_unchanged(self, model):
        """What the training loop actually consumes: one loss over the buffer."""
        rng = np.random.default_rng(47)
        graphs = [_random_graph(rng, n) for n in (14, 6, 21, 9)]
        md = torch.tensor(rng.standard_normal(1024).astype(np.float32))
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        self._no_dropout(model).train()
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.3))

        model.zero_grad()
        l_single = loss_fn(torch.stack(_single_forward(model, graphs, md)), target)
        l_single.backward()
        grads = {n: p.grad.detach().clone()
                 for n, p in model.named_parameters() if p.grad is not None}

        model.zero_grad()
        X, EI, centres = _collate(graphs)
        l_batched = loss_fn(torch.stack(_batched_forward(model, X, EI, centres, md)),
                            target)
        l_batched.backward()

        assert l_single.item() == pytest.approx(l_batched.item(), abs=1e-6)
        for n, p in model.named_parameters():
            if p.grad is not None:
                torch.testing.assert_close(p.grad, grads[n], rtol=1e-4, atol=1e-6,
                                           msg=f"gradient mismatch for {n}")


class TestReleasedForwardAcceptsBatches:
    """Everything above specifies the change; this exercises the real forward.

    `_batched_forward` is a reimplementation, so on its own it proves only that
    the MATHS is sound. These tests run `GAT_mut_processor.forward` itself with a
    tensor of sites and require it to reproduce the one-at-a-time answers, and
    require the int path to be untouched.
    """

    def test_tensor_of_sites_reproduces_the_per_sample_answers(self, model):
        rng = np.random.default_rng(53)
        graphs = [_random_graph(rng, n) for n in (11, 27, 5, 16)]
        md = torch.tensor(rng.standard_normal(1024).astype(np.float32))
        with torch.no_grad():
            single = _single_forward(model, graphs, md)
            X, EI, centres = _collate(graphs)
            out = model(X, EI, torch.tensor(centres), None,
                        md.unsqueeze(0).repeat(len(graphs), 1))
        assert out.shape == (len(graphs), 1)
        for a, b in zip(single, out.squeeze(-1)):
            assert a.item() == pytest.approx(b.item(), abs=1e-6)

    def test_per_sample_mutation_diffs_are_kept_with_their_own_graph(self, model):
        """Each row must pair with ITS graph -- a transposition would be silent."""
        rng = np.random.default_rng(59)
        graphs = [_random_graph(rng, n) for n in (9, 31, 14)]
        mds = [torch.tensor(rng.standard_normal(1024).astype(np.float32))
               for _ in graphs]
        with torch.no_grad():
            single = [model(torch.tensor(x), torch.tensor(ei), k, m).squeeze()
                      for (x, ei, k), m in zip(graphs, mds)]
            X, EI, centres = _collate(graphs)
            out = model(X, EI, torch.tensor(centres), None,
                        torch.stack(mds)).squeeze(-1)
            shuffled = model(X, EI, torch.tensor(centres), None,
                             torch.stack(mds[::-1])).squeeze(-1)
        for a, b in zip(single, out):
            assert a.item() == pytest.approx(b.item(), abs=1e-6)
        assert any(a.item() != pytest.approx(b.item(), abs=1e-6)
                   for a, b in zip(single, shuffled)), "diffs are interchangeable"

    @pytest.mark.parametrize("idx", [0, 3, 7])
    def test_the_int_path_is_unchanged(self, model, idx):
        """The released call signature must hit the same slice it always did."""
        rng = np.random.default_rng(61)
        x, ei, _ = _random_graph(rng, 9)
        md = torch.tensor(rng.standard_normal(1024).astype(np.float32))
        with torch.no_grad():
            out = model(torch.tensor(x), torch.tensor(ei), idx, md)
            h = torch.relu(model.complex_gat2(
                torch.relu(model.complex_gat1(torch.tensor(x), torch.tensor(ei))),
                torch.tensor(ei)))
            ref = model.binding_predictor(torch.cat(
                [h[idx:idx + 1], model.mutation_diff_processor(md.unsqueeze(0))],
                dim=-1))
        assert out.shape == (1, 1)
        assert out.item() == pytest.approx(ref.item(), abs=1e-7)

    def test_a_zero_dim_tensor_index_still_takes_the_scalar_path(self, model):
        """`train_fold` can pass a numpy/torch scalar; it must not become a batch."""
        rng = np.random.default_rng(67)
        x, ei, _ = _random_graph(rng, 9)
        md = torch.tensor(rng.standard_normal(1024).astype(np.float32))
        with torch.no_grad():
            a = model(torch.tensor(x), torch.tensor(ei), 4, md)
            b = model(torch.tensor(x), torch.tensor(ei), torch.tensor(4), md)
        assert a.shape == b.shape == (1, 1)
        assert a.item() == pytest.approx(b.item(), abs=1e-7)


class TestAblationModelsAgree:
    """Fig S4's ablation arms share the checkpoint format and must batch too."""

    def test_no_mut_ablation_batches(self):
        from model import GAT_mut_processor_no_mut
        torch.manual_seed(3)
        m = GAT_mut_processor_no_mut(D).eval()
        rng = np.random.default_rng(71)
        graphs = [_random_graph(rng, n) for n in (10, 22, 7)]
        with torch.no_grad():
            single = [m(torch.tensor(x), torch.tensor(ei), k).squeeze()
                      for x, ei, k in graphs]
            X, EI, centres = _collate(graphs)
            out = m(X, EI, torch.tensor(centres)).squeeze(-1)
        for a, b in zip(single, out):
            assert a.item() == pytest.approx(b.item(), abs=1e-6)

    def test_no_gat_ablation_batches(self):
        """Ignores the graph entirely, so the batch must come from the diffs."""
        from model import GAT_mut_processor_no_gat
        torch.manual_seed(4)
        m = GAT_mut_processor_no_gat().eval()
        rng = np.random.default_rng(73)
        graphs = [_random_graph(rng, n) for n in (10, 22, 7)]
        mds = [torch.tensor(rng.standard_normal(1024).astype(np.float32))
               for _ in graphs]
        with torch.no_grad():
            single = [m(torch.tensor(x), torch.tensor(ei), k, md).squeeze()
                      for (x, ei, k), md in zip(graphs, mds)]
            X, EI, centres = _collate(graphs)
            out = m(X, EI, torch.tensor(centres), None,
                    torch.stack(mds)).squeeze(-1)
        for a, b in zip(single, out):
            assert a.item() == pytest.approx(b.item(), abs=1e-6)
