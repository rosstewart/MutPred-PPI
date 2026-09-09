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
"""
import os

METHOD_DISPLAY_NAMES = {
    # MutPred-PPI (new megascale_all model)
    'MutPredPPI_sahni_megascale_all':                              'MutPred-PPI',
    'MutPredPPI_sahni_fragoza_megascale_all':                      'MutPred-PPI',
    'MutPredPPI_sahni_fragoza_varchamp1p_cava_megascale_all':      'MutPred-PPI',
    # SWING (both variants shown)
    'SWING_sahni_test_pretrain':                         'SWING (Test Pretrain)',
    'SWING_sahni_fragoza_test_pretrain':                 'SWING (Test Pretrain)',
    'SWING_sahni_fragoza_varchamp1p_cava_test_pretrain': 'SWING (Test Pretrain)',
    'SWING_sahni_no_test_pretrain':                      'SWING (Blind-Test)',
    'SWING_sahni_fragoza_no_test_pretrain':              'SWING (Blind-Test)',
    'SWING_sahni_fragoza_varchamp1p_cava_no_test_pretrain': 'SWING (Blind-Test)',
    # eSIG-Net
    'ESigNet_sahni':                                     'eSIG-Net',
    'ESigNet_sahni_fragoza':                             'eSIG-Net',
    'ESigNet_sahni_fragoza_varchamp1p_cava':             'eSIG-Net',
    # MINT
    'MINT_seq_diff_sahni':                               'MINT (seq diff)',
    'MINT_seq_diff_sahni_fragoza':                       'MINT (seq diff)',
    'MINT_seq_diff_sahni_fragoza_varchamp1p_cava':       'MINT (seq diff)',
    'MINT_site_diff_sahni':                              'MINT (site diff)',
    'MINT_site_diff_sahni_fragoza':                      'MINT (site diff)',
    'MINT_site_diff_sahni_fragoza_varchamp1p_cava':      'MINT (site diff)',
    # PPLM
    'PPLM_seq_diff_sahni':                               'PPLM (seq diff)',
    'PPLM_seq_diff_sahni_fragoza':                       'PPLM (seq diff)',
    'PPLM_seq_diff_sahni_fragoza_varchamp1p_cava':       'PPLM (seq diff)',
    'PPLM_site_diff_sahni':                              'PPLM (site diff)',
    'PPLM_site_diff_sahni_fragoza':                      'PPLM (site diff)',
    'PPLM_site_diff_sahni_fragoza_varchamp1p_cava':      'PPLM (site diff)',
    # varchamp2026 dataset
    'MutPredPPI_sahni_fragoza_varchamp2026_megascale_all': 'MutPred-PPI',
    'SWING_sahni_fragoza_varchamp2026_test_pretrain':    'SWING (Test Pretrain)',
    'SWING_sahni_fragoza_varchamp2026_no_test_pretrain': 'SWING (Blind-Test)',
    'ESigNet_sahni_fragoza_varchamp2026':                'eSIG-Net',
    'MINT_seq_diff_sahni_fragoza_varchamp2026':          'MINT (seq diff)',
    'MINT_site_diff_sahni_fragoza_varchamp2026':         'MINT (site diff)',
    'PPLM_seq_diff_sahni_fragoza_varchamp2026':          'PPLM (seq diff)',
    'PPLM_site_diff_sahni_fragoza_varchamp2026':         'PPLM (site diff)',
    # SFVCFP (sahni_fragoza_varchamp_full_pooled) dataset — S-new figure
    'MutPredPPI_sahni_fragoza_varchamp_full_pooled_megascale_all': 'MutPred-PPI',
    'SWING_sahni_fragoza_varchamp_full_pooled_test_pretrain':      'SWING (Test Pretrain)',
    'SWING_sahni_fragoza_varchamp_full_pooled_no_test_pretrain':   'SWING (Blind-Test)',
    'ESigNet_sahni_fragoza_varchamp_full_pooled':                  'eSIG-Net',
    'MINT_seq_diff_sahni_fragoza_varchamp_full_pooled':            'MINT (seq diff)',
    'MINT_site_diff_sahni_fragoza_varchamp_full_pooled':           'MINT (site diff)',
    'PPLM_seq_diff_sahni_fragoza_varchamp_full_pooled':            'PPLM (seq diff)',
    'PPLM_site_diff_sahni_fragoza_varchamp_full_pooled':           'PPLM (site diff)',
    # Fixed-predictor baselines
    'mutpred2_sahni':                                    'MutPred2',
    'saambe_sahni':                                      'SAAMBE-3D',
    'mutpred2_sahni_fragoza':                            'MutPred2',
    'saambe_sahni_fragoza':                              'SAAMBE-3D',
    'mutpred2_sahni_fragoza_varchamp1p_cava':            'MutPred2',
    'saambe_sahni_fragoza_varchamp1p_cava':              'SAAMBE-3D',
}


def extract_method_and_dataset(filename):
    """Extract method key and dataset from a comparison-mode results filename.

    Returns (method_key, dataset) where method_key matches a key in
    METHOD_DISPLAY_NAMES, or (None, dataset) for unrecognized files
    (ablation variants, novel experiments, etc.).
    """
    basename = os.path.basename(filename).replace('_detailed_results.pkl', '')

    if 'sahni_fragoza_varchamp1p_cava' in basename:
        dataset = 'sahni_fragoza_varchamp1p_cava'
    elif 'sahni_fragoza_varchamp2026' in basename:
        dataset = 'sahni_fragoza_varchamp2026'
    elif 'sahni_fragoza_varchamp_full_pooled' in basename:
        dataset = 'sahni_fragoza_varchamp_full_pooled'
    elif 'sahni_fragoza_varchamp_pooled' in basename:
        dataset = 'sahni_fragoza_varchamp_pooled'
    elif 'sahni_fragoza_varchamp_full' in basename:
        dataset = 'sahni_fragoza_varchamp_full'
    elif 'sahni_varchamp1p_cava' in basename:
        dataset = 'sahni_varchamp1p_cava'
    elif 'sahni_fragoza' in basename:
        dataset = 'sahni_fragoza'
    elif 'sahni' in basename:
        dataset = 'sahni'
    else:
        dataset = 'unknown'

    if basename == f'MutPredPPI_{dataset}_megascale_all':
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
    "ddmutppi_":            "DDMutPPI",
}


def with_baseline_variants(datasets):
    """METHOD_DISPLAY_NAMES plus per-dataset baseline keys, as a NEW dict."""
    out = dict(METHOD_DISPLAY_NAMES)
    for prefix, label in _BASELINE_PREFIXES.items():
        out.update({f"{prefix}{d}": label for d in datasets})
    return out
