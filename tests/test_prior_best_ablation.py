"""The Fig S4 'Prior Best' arm must reproduce the prior model's own recipe.

'Prior Best' is the previously published model (RECOMB 2024 conference version)
re-run as a fine-tuning arm. Its value as a baseline depends entirely on being
fine-tuned the way it originally was, so the freeze policy is pinned here
against the original code rather than described in prose:

    for param in model.parameters():                       param.requires_grad = False
    for param in model.mutation_diff_processor[-1].parameters(): param.requires_grad = True
    for param in model.binding_predictor.parameters():      param.requires_grad = True
    for param in model.complex_gat1.parameters():           param.requires_grad = True
    for param in model.complex_gat2.parameters():           param.requires_grad = True

Only the FIRST Linear of the mutation-diff processor stays frozen. This is the
`full`/`megascale` policy, NOT `freeze_mut_processor`, which additionally
freezes the last Linear and so trains a strictly smaller set.
"""
import torch
import pytest

from model import GAT_mut_processor, apply_freeze_strategy


# loads the archived prior-best checkpoint; see tests/conftest.py for the opt-in flags.
pytestmark = pytest.mark.requires_data


def _reference_recipe(model):
    """The original code, verbatim in behaviour."""
    for p in model.parameters():
        p.requires_grad = False
    for group in (model.mutation_diff_processor[-1], model.binding_predictor,
                  model.complex_gat1, model.complex_gat2):
        for p in group.parameters():
            p.requires_grad = True
    return model


def _split(model):
    tr = {n for n, p in model.named_parameters() if p.requires_grad}
    fz = {n for n, p in model.named_parameters() if not p.requires_grad}
    return tr, fz


class TestFreezePolicy:
    def test_matches_the_original_recipe_exactly(self):
        want_tr, want_fz = _split(_reference_recipe(GAT_mut_processor(1024)))
        got_tr, got_fz = _split(apply_freeze_strategy(GAT_mut_processor(1024),
                                                     "prior_best"))
        assert got_tr == want_tr
        assert got_fz == want_fz

    def test_only_the_first_linear_of_the_diff_processor_is_frozen(self):
        _tr, fz = _split(apply_freeze_strategy(GAT_mut_processor(1024), "prior_best"))
        assert fz == {"mutation_diff_processor.0.weight",
                      "mutation_diff_processor.0.bias"}

    def test_both_gats_and_the_head_train(self):
        tr, _fz = _split(apply_freeze_strategy(GAT_mut_processor(1024), "prior_best"))
        for prefix in ("complex_gat1", "complex_gat2", "binding_predictor",
                       "mutation_diff_processor.3"):
            assert any(n.startswith(prefix) for n in tr), prefix

    def test_it_is_not_the_freeze_diff_policy(self):
        """Negative control -- these two arms must differ.

        `freeze_mut_processor` also freezes the LAST Linear, so it trains a
        strictly smaller set. Confusing the two would silently understate the
        prior model.
        """
        pb, _ = _split(apply_freeze_strategy(GAT_mut_processor(1024), "prior_best"))
        fd, _ = _split(apply_freeze_strategy(GAT_mut_processor(1024),
                                             "freeze_mut_processor"))
        assert fd < pb, "freeze_diff should train a strict subset of prior_best"
        assert pb - fd == {"mutation_diff_processor.3.weight",
                           "mutation_diff_processor.3.bias"}


class TestArtifacts:
    def test_checkpoint_and_scaler_are_archived_not_shipped(self):
        """They must not sit in weights/, where a production run could find them."""
        from training.train_fold import (_PRIOR_BEST_PRETRAINED_PATH,
                                         _PRIOR_BEST_SCALER_PATH)
        for p in (_PRIOR_BEST_PRETRAINED_PATH, _PRIOR_BEST_SCALER_PATH):
            assert p.exists(), p
            assert "archive" in p.parts, f"{p} should live under archive/"
            assert "weights" not in p.parts

    def test_checkpoint_loads_completely(self):
        """Every tensor transfers -- unlike the monomer stability pretrains."""
        from training.train_fold import _PRIOR_BEST_PRETRAINED_PATH
        sd = torch.load(_PRIOR_BEST_PRETRAINED_PATH, map_location="cpu",
                        weights_only=False)
        sd = sd.get("state_dict", sd) if "complex_gat1.lin.weight" not in sd else sd
        cur = GAT_mut_processor(1024).state_dict()
        assert set(sd) == set(cur)
        for k in sd:
            assert tuple(sd[k].shape) == tuple(cur[k].shape), k

    def test_scaler_matches_the_diff_dimension(self):
        import joblib
        from training.train_fold import _PRIOR_BEST_SCALER_PATH
        assert joblib.load(_PRIOR_BEST_SCALER_PATH).n_features_in_ == 1024
