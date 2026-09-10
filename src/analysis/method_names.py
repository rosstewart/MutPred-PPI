"""Canonical method-key -> display-name mapping, and the filename parser.

`roc_plots.py` and `export_reconstruction_tables.py` each carried their own copy.
They had drifted: the dicts were 46 vs 40 keys (roc_plots a strict superset, no
conflicting values) and the two `extract_method_and_dataset` bodies were 59 vs 74
lines despite the second one's docstring claiming it was copied verbatim. Verified
before merging: the functions agree on all 63 real `*_detailed_results.pkl`
filenames, and the 6 extra keys match zero files on disk, so unifying on the
superset changes no output today.

Both consumers use the mapping as a membership filter, and `roc_plots` used to
extend it by mutating the module-level dict in place -- which would leak across
consumers once shared. Use `with_baseline_variants()` instead; it returns a new
dict.

**Canonical datasets only (2026-09-10).** The three GCV datasets any
comparison figure reads (Fig 3, S1, S7) are `sahni_only_mapped090826`,
`sahni_fragoza_mapped090826`, `sahni_fragoza_varchamp_all_mapped090826` --
see `utils.gcv_common.DATASET_CONFIGS` and `roc_plots.CANONICAL_DATASETS`.
`_SHORT_DATASET_NAMES` is the one place that maps a full config name to the
short display key (`sahni`, `sahni_fragoza`, `sahni_fragoza_varchamp_all`)
used throughout `roc_plots.py` (SAAMBE test-class files, MutPred2-standalone
label files, `METHOD_DISPLAY_NAMES` keys) -- `roc_plots.CANONICAL_DATASETS`
must stay a re-expression of this same mapping, not a second copy of it.

Every prior dataset-name family this module used to recognize
(`sahni_fragoza_varchamp1p_cava`, `sahni_varchamp1p_cava`,
`sahni_fragoza_varchamp2026`, `sahni_fragoza_varchamp_full[_pooled]`,
`sahni_fragoza_varchamp_pooled`) named datasets that no longer exist, or exist
under a different row ordering. A GCV result file whose name lacks a
`_mapped090826` suffix predates the 090826 rebaseline outright -- e.g. the
`MutPredPPI_sahni_fragoza_megascale_all_detailed_results.pkl` files currently
on disk in `results/gcv/` are pre-090826 (a fresh run writes
`MutPredPPI_sahni_fragoza_mapped090826_megascale_all_...`, per
`mutpred_ppi_gcv.py`'s `result_stem=f"MutPredPPI_{cfg.name}_{args.ablation}"`
where `cfg.name` is the full `DATASET_CONFIGS` key). `extract_method_and_dataset`
returns `dataset='legacy'` for both cases so the caller can flag it instead of
silently plotting it alongside fresh results (see `utils.legacy_guard`).
"""
import os

from utils.legacy_guard import reject_legacy_dataset_name, LegacyInputError

# Full `DATASET_CONFIGS` key -> short display key. Order matters: the
# compound name must be checked before its `sahni_fragoza` prefix matches.
_SHORT_DATASET_NAMES = {
    "sahni_fragoza_varchamp_all_mapped090826": "sahni_fragoza_varchamp_all",
    "sahni_fragoza_mapped090826":              "sahni_fragoza",
    "sahni_only_mapped090826":                 "sahni",
}

METHOD_DISPLAY_NAMES = {
    # MutPred-PPI (new megascale_all model)
    'MutPredPPI_sahni_megascale_all':                              'MutPred-PPI',
    'MutPredPPI_sahni_fragoza_megascale_all':                      'MutPred-PPI',
    'MutPredPPI_sahni_fragoza_varchamp_all_megascale_all':         'MutPred-PPI',
    # SWING (both variants shown)
    'SWING_sahni_test_pretrain':                              'SWING (Test Pretrain)',
    'SWING_sahni_fragoza_test_pretrain':                      'SWING (Test Pretrain)',
    'SWING_sahni_fragoza_varchamp_all_test_pretrain':         'SWING (Test Pretrain)',
    'SWING_sahni_no_test_pretrain':                           'SWING (Blind-Test)',
    'SWING_sahni_fragoza_no_test_pretrain':                   'SWING (Blind-Test)',
    'SWING_sahni_fragoza_varchamp_all_no_test_pretrain':      'SWING (Blind-Test)',
    # eSIG-Net
    'ESigNet_sahni':                                          'eSIG-Net',
    'ESigNet_sahni_fragoza':                                  'eSIG-Net',
    'ESigNet_sahni_fragoza_varchamp_all':                     'eSIG-Net',
    # MINT
    'MINT_seq_diff_sahni':                                    'MINT (seq diff)',
    'MINT_seq_diff_sahni_fragoza':                             'MINT (seq diff)',
    'MINT_seq_diff_sahni_fragoza_varchamp_all':                'MINT (seq diff)',
    'MINT_site_diff_sahni':                                    'MINT (site diff)',
    'MINT_site_diff_sahni_fragoza':                             'MINT (site diff)',
    'MINT_site_diff_sahni_fragoza_varchamp_all':                'MINT (site diff)',
    # PPLM
    'PPLM_seq_diff_sahni':                                     'PPLM (seq diff)',
    'PPLM_seq_diff_sahni_fragoza':                              'PPLM (seq diff)',
    'PPLM_seq_diff_sahni_fragoza_varchamp_all':                 'PPLM (seq diff)',
    'PPLM_site_diff_sahni':                                    'PPLM (site diff)',
    'PPLM_site_diff_sahni_fragoza':                             'PPLM (site diff)',
    'PPLM_site_diff_sahni_fragoza_varchamp_all':                'PPLM (site diff)',
    # Fixed-predictor baselines (SAAMBE-3D/MutPred2 pretrained, scored not trained)
    'mutpred2_sahni':                     'MutPred2',
    'saambe_sahni':                       'SAAMBE-3D',
    'mutpred2_sahni_fragoza':             'MutPred2',
    'saambe_sahni_fragoza':               'SAAMBE-3D',
    'mutpred2_sahni_fragoza_varchamp_all': 'MutPred2',
    'saambe_sahni_fragoza_varchamp_all':   'SAAMBE-3D',
}


def extract_method_and_dataset(filename):
    """Extract method key and dataset from a comparison-mode results filename.

    Returns (method_key, dataset) where `dataset` is the SHORT display key
    (`sahni`, `sahni_fragoza`, `sahni_fragoza_varchamp_all` -- matching
    `roc_plots.CANONICAL_DATASETS`) and `method_key` matches a key in
    METHOD_DISPLAY_NAMES, or (None, dataset) for unrecognized files (ablation
    variants, novel experiments, etc.). `dataset='legacy'` specifically means
    the filename carries a retired dataset-name token, or lacks the
    `_mapped090826` suffix a fresh GCV run always writes -- callers should
    flag this distinctly from a generic "unrecognized" skip, since it means a
    stale file is sitting where a fresh one is expected.
    """
    basename = os.path.basename(filename).replace('_detailed_results.pkl', '')

    try:
        reject_legacy_dataset_name(basename)
    except LegacyInputError:
        return None, 'legacy'

    full_name = next((f for f in _SHORT_DATASET_NAMES if f in basename), None)
    if full_name is None:
        # No recognized _mapped090826 dataset fragment at all -- most likely a
        # pre-090826 file whose dataset name never picked up the suffix (e.g.
        # today's on-disk results/gcv/*_detailed_results.pkl). Still legacy,
        # not merely "unknown": every canonical GCV run writes one of the
        # three _mapped090826 fragments into its result_stem.
        return None, 'legacy'
    dataset = _SHORT_DATASET_NAMES[full_name]

    if basename == f'MutPredPPI_{full_name}_megascale_all':
        method = f'MutPredPPI_{dataset}_megascale_all'
    elif basename.startswith('gnn_'):
        # old GATMutPPI — superseded by MutPredPPI_*_megascale_all; skip in comparison
        return None, dataset
    elif 'SWING' in basename:
        if 'no_test_pretrain' in basename:
            method = 'SWING_' + dataset + '_no_test_pretrain'
        else:
            method = 'SWING_' + dataset + '_test_pretrain'
    elif basename.startswith('ESigNet_'):
        method = 'ESigNet_' + dataset
    elif basename.startswith('MINT_seq_diff_'):
        method = 'MINT_seq_diff_' + dataset
    elif basename.startswith('MINT_site_diff_'):
        method = 'MINT_site_diff_' + dataset
    elif basename.startswith('PPLM_seq_diff_'):
        method = 'PPLM_seq_diff_' + dataset
    elif basename.startswith('PPLM_site_diff_'):
        method = 'PPLM_site_diff_' + dataset
    elif basename.startswith('MonoIFaceHierSpurs_'):
        method = 'MonoIFaceHierSpurs_' + dataset
    elif basename.startswith('mutpred2_'):
        method = 'mutpred2_' + dataset
    elif basename.startswith('saambe_'):
        method = 'saambe_' + dataset
    else:
        # Not a recognized comparison method (e.g. MutPredPPI ablation variant)
        return None, dataset

    return method, dataset


# Zero-shot / external baselines are keyed per dataset and were previously added
# by mutating METHOD_DISPLAY_NAMES in place inside roc_plots.main().
_BASELINE_PREFIXES = {
    "mutpred2_standalone_": "MutPred2",
    "saambe_3d_":           "SAAMBE-3D",
    "mutppi_":              "MutPPI",
    "mutppiplus_":          "MutPPI+",
}


def with_baseline_variants(datasets):
    """METHOD_DISPLAY_NAMES plus per-dataset baseline keys, as a NEW dict.

    `datasets` are SHORT display keys (`sahni`, `sahni_fragoza`,
    `sahni_fragoza_varchamp_all`), matching `roc_plots.CANONICAL_DATASETS`'s
    keys -- these are not full `_mapped090826` config names, and are not
    validated against `utils.legacy_guard` for that reason (they never carry
    a legacy token; the caller derives them from `CANONICAL_DATASETS`, not
    from a filename).
    """
    out = dict(METHOD_DISPLAY_NAMES)
    for prefix, label in _BASELINE_PREFIXES.items():
        out.update({f"{prefix}{d}": label for d in datasets})
    return out
