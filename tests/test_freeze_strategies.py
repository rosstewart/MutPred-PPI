"""Which parameters each ablation arm trains.

Fig S4 reads as "how much of the pretrained model has to move?", so each arm is
only interpretable if its freeze policy is exactly what its name claims. These
are cheap to get subtly wrong -- `freeze_mut_processor` and `prior_best` differ
by a single Linear -- and wrong in a way no AUC would reveal, so the policies are
pinned as parameter sets rather than described in prose.

`freeze_mut_processor` and `freeze_gat` are complements: the first fixes the
mutation representation and adapts the structural half, the second does the
reverse. Together they bracket which half fine-tuning actually needs.
"""
import pytest

from model import GAT_mut_processor, apply_freeze_strategy

MUT = "mutation_diff_processor"
GAT = ("complex_gat1", "complex_gat2")


def _frozen(ablation):
    m = apply_freeze_strategy(GAT_mut_processor(1024), ablation)
    return {n for n, p in m.named_parameters() if not p.requires_grad}


def _trainable(ablation):
    m = apply_freeze_strategy(GAT_mut_processor(1024), ablation)
    return {n for n, p in m.named_parameters() if p.requires_grad}


class TestFreezeMutProcessor:
    def test_freezes_the_entire_mutation_processor(self):
        assert _frozen("freeze_mut_processor") == {
            f"{MUT}.0.weight", f"{MUT}.0.bias", f"{MUT}.3.weight", f"{MUT}.3.bias"}

    def test_trains_both_gats_and_the_head(self):
        tr = _trainable("freeze_mut_processor")
        for pre in GAT + ("binding_predictor",):
            assert any(n.startswith(pre) for n in tr), pre


class TestFreezeGat:
    def test_freezes_both_gats_and_nothing_else(self):
        fz = _frozen("freeze_gat")
        assert fz and all(n.startswith(GAT) for n in fz)

    def test_trains_the_whole_mutation_processor_and_the_head(self):
        tr = _trainable("freeze_gat")
        for n in (f"{MUT}.0.weight", f"{MUT}.3.weight",
                  "binding_predictor.0.weight", "binding_predictor.3.weight"):
            assert n in tr, n

    def test_it_is_the_complement_of_freeze_mut_processor(self):
        """The two arms partition the non-head parameters.

        The head is NOT part of the partition: `binding_predictor` trains in both
        arms, because a frozen head would make the two arms measure something
        else entirely (whether the head alone can compensate) rather than which
        representation needs to move.
        """
        a, b = _frozen("freeze_mut_processor"), _frozen("freeze_gat")
        allp = _frozen("freeze_gat") | _trainable("freeze_gat")
        body = {n for n in allp if not n.startswith("binding_predictor")}
        assert a & b == set(), "an arm freezes something the other also freezes"
        assert a | b == body, "a non-head parameter is frozen by neither arm"

    def test_the_head_trains_in_both_arms(self):
        for arm in ("freeze_mut_processor", "freeze_gat"):
            assert not any(n.startswith("binding_predictor")
                           for n in _frozen(arm)), arm


class TestArmsAreDistinct:
    @pytest.mark.parametrize("a,b", [
        ("freeze_mut_processor", "freeze_gat"),
        ("freeze_mut_processor", "prior_best"),
        ("freeze_gat", "prior_best"),
        ("freeze_mut_processor", "megascale_head"),
    ])
    def test_no_two_arms_share_a_policy(self, a, b):
        """A rename that silently collapsed two arms would show up here."""
        assert _trainable(a) != _trainable(b), f"{a} and {b} train the same set"

    def test_head_only_is_the_smallest(self):
        head = _trainable("megascale_head")
        for other in ("freeze_mut_processor", "freeze_gat", "prior_best",
                      "megascale_all"):
            assert head < _trainable(other) or head == _trainable(other), other

    def test_megascale_all_trains_everything(self):
        assert _frozen("megascale_all") == set()

    def test_the_retired_name_is_gone(self):
        """`megascale_freeze_diff` was renamed; it must not silently no-op.

        `apply_freeze_strategy` falls through to "train everything" for an
        unrecognised name, so a stale caller would quietly get megascale_all
        rather than an error. This records that the old name now means that.
        """
        assert _frozen("megascale_freeze_diff") == set()


class TestAblationFigureNaming:
    """The three freeze arms must read as one set in the figure.

    `megascale_head` trains only `binding_predictor`, so BOTH the mutation
    processor and both GATs are held fixed -- it is the third member of the
    freeze family, not a separate idea. It used to be labelled "Head Only",
    which named what stayed trainable while its two siblings named what was
    frozen.
    """

    def test_freeze_family_is_named_consistently(self):
        from analysis.roc_plots import ABLATION_DISPLAY_NAMES
        names = set(ABLATION_DISPLAY_NAMES.values())
        assert {"Freeze Mut Processor", "Freeze GAT", "Freeze Both"} <= names
        assert "Head Only" not in names

    def test_every_display_name_has_a_colour(self):
        from analysis.roc_plots import ABLATION_COLORS, ABLATION_DISPLAY_NAMES
        missing = set(ABLATION_DISPLAY_NAMES.values()) - set(ABLATION_COLORS)
        assert not missing, f"ablation arms with no colour: {missing}"


class TestPriorBestIsOptIn:
    """The previously published model is not an ablation of this architecture.

    Its checkpoint ships in neither the repository nor the Zenodo deposit, so
    drawing it by default leaves an external reproducer with a bar they cannot
    regenerate and no explanation.
    """

    def test_off_by_default(self):
        from analysis import roc_plots
        assert roc_plots.INCLUDE_PRIOR_BEST is False

    def test_runner_defaults_to_excluding_it(self):
        import inspect

        from analysis import run_roc_ablation
        sig = inspect.signature(run_roc_ablation.main)
        assert sig.parameters["include_prior_best"].default is False

    def test_the_arm_is_still_registered(self):
        """Excluded by default, but still plottable on request."""
        from analysis.roc_plots import (ABLATION_DISPLAY_NAMES,
                                        PRIOR_BEST_DISPLAY_NAME)
        assert PRIOR_BEST_DISPLAY_NAME in ABLATION_DISPLAY_NAMES.values()
