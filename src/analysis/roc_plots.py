# %% [markdown]
# # ROC Plots with GCV Iterations

# %% Imports
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import glob
import os

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
from method_names import (  # noqa: E402
    METHOD_DISPLAY_NAMES, extract_method_and_dataset, with_baseline_variants)
from paths import ANNOTATIONS_DIR, CV_DIR, GCV_RESULTS_DIR  # noqa: E402
from ids import get_gene_name, is_uniprot_accession, split_wt_id  # noqa: E402  (single definition, see src/ids.py)
from gcv_curves import N_SEM_DIVISOR  # noqa: E402  (single definition)


WORKING_DIR = str(GCV_RESULTS_DIR)
# Canonical CV artifacts, in-repo (datasets/cv_reference/).  These were
# hardcoded to an external cv_splits/ whose seed-1 pair_test_classes is the
# corrupt file of defect 2 -- so the iptm stratification silently missed that
# fix while every GNN method received it.  See docs/METHOD_PROVENANCE.md.
CV_REF = str(CV_DIR)
# %% Helper functions
with open(ANNOTATIONS_DIR / 'all_to_uniprot.pkl', 'rb') as f:
    all_to_uniprot = pickle.load(f)

with open(ANNOTATIONS_DIR / 'confidence_scores.pkl', 'rb') as f:
    iptm_scores_raw = pickle.load(f)

SAAMBE_train_uniprots = np.load(f'{WORKING_DIR}/SAAMBE_train_uniprots.npy')

from collections import defaultdict


iptm_scores = defaultdict(dict)
for key in iptm_scores_raw:
    p1, p2 = split_wt_id(key)
    if not (p1.startswith('NP_') or p1.startswith('np_')):
        p1 = p1.replace('_', '-')
    p1 = p1.upper()
    p2 = p2.replace('_', '-').upper()
    if not is_uniprot_accession(p1) and not is_uniprot_accession(p2) and not (p1 in all_to_uniprot and p2 in all_to_uniprot):
        print(p1, p2)
    if not is_uniprot_accession(p1) and p1 in all_to_uniprot:
        p1 = all_to_uniprot[p1]
    if not is_uniprot_accession(p2) and p2 in all_to_uniprot:
        p2 = all_to_uniprot[p2]
    if '-' in p1 and f'{p1.split("-")[0]}|{p2}' not in iptm_scores:
        iptm_scores[f'{p1.split("-")[0]}|{p2}']['iptm'] = iptm_scores_raw[key]['iptm']
        iptm_scores[f'{p1.split("-")[0]}|{p2}']['ptm'] = iptm_scores_raw[key]['ptm']
    if '-' in p2 and f'{p1}|{p2.split("-")[0]}' not in iptm_scores:
        iptm_scores[f'{p1}|{p2.split("-")[0]}']['iptm'] = iptm_scores_raw[key]['iptm']
        iptm_scores[f'{p1}|{p2.split("-")[0]}']['ptm'] = iptm_scores_raw[key]['ptm']
    iptm_scores[f'{p1}|{p2}']['iptm'] = iptm_scores_raw[key]['iptm']
    iptm_scores[f'{p1}|{p2}']['ptm'] = iptm_scores_raw[key]['ptm']

datasets = ('sahni', 'sahni_fragoza', 'sahni_varchamp1p_cava')
prefixes = ('', 'sahni_fragoza_train_', 'sahni_varchamp1p_cava_train_')
pair_test_prefixes = ('', 'swing_train_', 'combined_sahni_varchamp1p_cava_seq_confirmed_concat_clust_')
labels_txt_fs = ['all_vt_ids_and_labels.txt',
                 'sahni_fragoza_all_vt_ids_and_labels.txt',
                 'combined_sahni_varchamp1p_cava_seq_confirmed_all_vt_ids_and_labels.txt']
dataset_labels = [[], [], []]
dataset_proteins = [[], [], []]
dataset_complexes = [[], [], []]
dataset_iptms = [[], [], []]
dataset_ptms = [[], [], []]
dataset_saambe_test_classes = [[], [], []]

for i, dataset in enumerate(datasets):
    with open(f'{CV_REF}/{labels_txt_fs[i]}', 'r') as f:
        for line in f:
            complex_id, variant, label = line.strip().split(' ')
            dataset_labels[i].append(int(label))

            part_1, part_2 = split_wt_id(complex_id)
            part_1_added, part_2_added = False, False
            if is_uniprot_accession(part_1):
                part_1_added = True
            if is_uniprot_accession(part_2):
                part_2_added = True

            part_1, part_2 = get_gene_name(part_1), get_gene_name(part_2)
            if not part_1_added:
                part_1 = all_to_uniprot[part_1] if part_1 in all_to_uniprot else 'XXX'
            if not part_2_added:
                part_2 = all_to_uniprot[part_2] if part_2 in all_to_uniprot else 'XXX'

            dataset_proteins[i].append(part_1)
            dataset_proteins[i].append(part_2)
            mapped_complex = f'{part_1}|{part_2}'
            mapped_flipped_complex = f'{part_2}|{part_1}'
            dataset_complexes[i].append(mapped_complex)
            if mapped_complex in iptm_scores:
                dataset_iptms[i].append(iptm_scores[mapped_complex]['iptm'])
                dataset_ptms[i].append(iptm_scores[mapped_complex]['ptm'])
            elif mapped_flipped_complex in iptm_scores:
                dataset_iptms[i].append(iptm_scores[mapped_flipped_complex]['iptm'])
                dataset_ptms[i].append(iptm_scores[mapped_flipped_complex]['ptm'])
            else:
                dataset_iptms[i].append(float('nan'))
                dataset_ptms[i].append(float('nan'))

            if part_1 in SAAMBE_train_uniprots and part_2 in SAAMBE_train_uniprots:
                dataset_saambe_test_classes[i].append(1)
            elif part_1 in SAAMBE_train_uniprots or part_2 in SAAMBE_train_uniprots:
                dataset_saambe_test_classes[i].append(2)
            else:
                dataset_saambe_test_classes[i].append(3)

    dataset_labels[i] = np.array(dataset_labels[i])
    dataset_proteins[i] = np.array(dataset_proteins[i])
    dataset_saambe_test_classes[i] = np.array(dataset_saambe_test_classes[i])

for i, dataset_arr in enumerate(dataset_proteins):
    print(datasets[i], sum(dataset_arr == 'XXX'), 'non-mappable, ', len(dataset_arr), 'total,')
    for class_ in (1, 2, 3):
        print('\t', len(dataset_saambe_test_classes[i][dataset_saambe_test_classes[i] == class_]))
    np.save(f'{WORKING_DIR}/{datasets[i]}_SAAMBE-3D_test_classes.npy', dataset_saambe_test_classes[i])

# %% Build GCV split pkl files
for i, dataset in enumerate(datasets):

    _base_splits_path = f'{CV_REF}/{prefixes[i]}fold_splits.pkl'
    if not os.path.exists(_base_splits_path):
        print(f'Warning: {_base_splits_path} not found — skipping iptm/saambe processing for {dataset}')
        continue

    with open(_base_splits_path, 'rb') as f:
        base_fold_splits = pickle.load(f)

    mutpred2_preds_exist, saambe_preds_exist = 1, 1
    if os.path.exists(f'{WORKING_DIR}/{dataset}_mutpred2_standalone_preds.npy'):
        mutpred2_preds_1 = np.load(f'{WORKING_DIR}/{dataset}_mutpred2_standalone_preds.npy')
    else:
        mutpred2_preds_1 = np.array([0.5] * len(dataset_labels[i]))
        mutpred2_preds_exist = 0

    if os.path.exists(f'{WORKING_DIR}/{dataset}_SAAMBE-3D_preds.npy'):
        saambe_preds_1 = np.load(f'{WORKING_DIR}/{dataset}_SAAMBE-3D_preds.npy')
    else:
        saambe_preds_1 = np.array([0.5] * len(dataset_labels[i]))
        saambe_preds_exist = 0

    iptms_1     = np.array(dataset_iptms[i])
    ptms_1      = np.array(dataset_ptms[i])
    complexes_1 = np.array(dataset_complexes[i])

    mutpred2_labels_1 = dataset_labels[i]

    print(dataset, len(mutpred2_labels_1))

    assert (len(mutpred2_preds_1) == len(saambe_preds_1) ==
            len(mutpred2_labels_1)), \
        f'{len(mutpred2_preds_1)} {len(saambe_preds_1)} {len(mutpred2_labels_1)}'

    mutpred2_base_preds  = np.empty(len(mutpred2_preds_1))
    mutpred2_base_labels = np.empty(len(mutpred2_preds_1))
    saambe_base_preds    = np.empty(len(mutpred2_preds_1))
    base_iptms           = np.empty(len(iptms_1))
    base_ptms            = np.empty(len(ptms_1))
    base_complexes       = np.empty(len(complexes_1), dtype=object)

    curr_idx = 0
    for fold, train_idx, test_idx in base_fold_splits:
        for idx in test_idx:
            mutpred2_base_preds[idx]  = mutpred2_preds_1[curr_idx]
            mutpred2_base_labels[idx] = mutpred2_labels_1[curr_idx]
            saambe_base_preds[idx]    = saambe_preds_1[curr_idx]
            base_iptms[idx]           = iptms_1[curr_idx]
            base_ptms[idx]            = ptms_1[curr_idx]
            base_complexes[idx]       = complexes_1[curr_idx]
            curr_idx += 1

    mutpred2_detailed_results = {'iterations': {}}
    saambe_detailed_results   = {'iterations': {}}
    iptm_detailed_results     = {'iterations': {}}

    for gcv_iter in range(30):
        with open(f'{CV_REF}/{prefixes[i]}fold_splits_{gcv_iter}.pkl', 'rb') as f:
            fold_splits = pickle.load(f)

        split_pair_test_classes = np.load(
            f'{CV_REF}/'
            f'{pair_test_prefixes[i]}pair_test_classes_{gcv_iter}.npy'
        )
        fold_n_test = []

        mutpred2_split_preds  = []
        mutpred2_split_labels = []
        saambe_split_preds    = []
        split_iptms           = []
        split_ptms            = []
        split_complexes       = []

        for fold, train_idx, test_idx in fold_splits:
            fold_n_test.append(len(test_idx))
            mutpred2_split_preds.extend(mutpred2_base_preds[test_idx])
            mutpred2_split_labels.extend(mutpred2_base_labels[test_idx])
            saambe_split_preds.extend(saambe_base_preds[test_idx])
            split_iptms.extend(base_iptms[test_idx])
            split_ptms.extend(base_ptms[test_idx])
            split_complexes.extend(base_complexes[test_idx])

        mutpred2_split_preds  = np.array(mutpred2_split_preds)
        mutpred2_split_labels = np.array(mutpred2_split_labels)
        saambe_split_preds    = np.array(saambe_split_preds)
        split_iptms           = np.array(split_iptms)
        split_ptms            = np.array(split_ptms)
        split_complexes       = np.array(split_complexes, dtype=object)

        if gcv_iter == 1:
            assert (np.array_equal(mutpred2_split_preds,  mutpred2_preds_1) and
                    np.array_equal(mutpred2_split_labels, mutpred2_labels_1) and
                    np.array_equal(saambe_split_preds,    saambe_preds_1))

        mutpred2_iteration_results = {'folds': {}}
        saambe_iteration_results   = {'folds': {}}
        iptm_iteration_results     = {'folds': {}}

        curr_idx = 0

        for fold, n_test in enumerate(fold_n_test):
            mutpred2_preds_fold  = mutpred2_split_preds[curr_idx:curr_idx + n_test]
            mutpred2_labels_fold = mutpred2_split_labels[curr_idx:curr_idx + n_test]
            saambe_preds_fold    = saambe_split_preds[curr_idx:curr_idx + n_test]
            fold_iptms           = split_iptms[curr_idx:curr_idx + n_test]
            fold_ptms            = split_ptms[curr_idx:curr_idx + n_test]
            fold_complexes       = split_complexes[curr_idx:curr_idx + n_test]
            pair_test            = split_pair_test_classes[curr_idx:curr_idx + n_test]

            mutpred2_fold_results = {
                f'class_{c}': {'preds': [], 'labels': [], 'auc': None}
                for c in [1, 2, 3]
            }
            saambe_fold_results = {
                f'class_{c}': {'preds': [], 'labels': [], 'auc': None}
                for c in [1, 2, 3]
            }
            iptm_fold_results = {
                f'class_{c}': {'complex_ids': [], 'iptms': [], 'ptms': []}
                for c in [1, 2, 3]
            }

            for pair_test_class in (1, 2, 3):
                valid_mask = (pair_test == pair_test_class) & (mutpred2_preds_fold != -1)
                ck = f'class_{pair_test_class}'

                mutpred2_class_preds = mutpred2_preds_fold[valid_mask]
                saambe_class_preds   = saambe_preds_fold[valid_mask]
                class_iptms          = fold_iptms[valid_mask]
                class_ptms           = fold_ptms[valid_mask]
                class_complexes      = fold_complexes[valid_mask]
                class_labels         = mutpred2_labels_fold[valid_mask]

                mutpred2_fold_results[ck]['preds']   = mutpred2_class_preds.copy()
                mutpred2_fold_results[ck]['labels']  = class_labels.copy()
                saambe_fold_results[ck]['preds']     = saambe_class_preds.copy()
                saambe_fold_results[ck]['labels']    = class_labels.copy()
                iptm_fold_results[ck]['complex_ids'] = class_complexes.tolist()
                iptm_fold_results[ck]['iptms']       = class_iptms.copy()
                iptm_fold_results[ck]['ptms']        = class_ptms.copy()

                n_pos = np.sum(class_labels == 1)
                n_neg = np.sum(class_labels == 0)

                if n_pos > 0 and n_neg > 0:
                    mutpred2_fold_results[ck]['auc'] = roc_auc_score(class_labels, mutpred2_class_preds)
                    saambe_fold_results[ck]['auc']   = roc_auc_score(class_labels, saambe_class_preds)
                else:
                    mutpred2_fold_results[ck]['auc'] = np.nan
                    saambe_fold_results[ck]['auc']   = np.nan

            iptm_iteration_results['folds'][fold] = iptm_fold_results
            curr_idx += n_test

        iptm_detailed_results['iterations'][gcv_iter] = iptm_iteration_results

    with open(f'{WORKING_DIR}/iptm_{dataset}_gcv_splits.pkl', 'wb') as f:
        pickle.dump(iptm_detailed_results, f)

# %% [markdown]
# ### Main Analysis

# %% Configuration
SAVE_PLOTS = True
SAVE_DIR = "roc_plots_with_variance"

FOR_SLIDES = False
TITLE_FONTSIZE  = 20 if FOR_SLIDES else 14
FONTSIZE_LEGEND = 11 if FOR_SLIDES else 9
FONTSIZE_AXIS   = 16 if FOR_SLIDES else 12

# ── Toggle these to switch modes ─────────────────────────────────────────────
ABLATION = False   # True = ablation plots, False = method comparison plots
PRC      = False   # False → ROC curves / AUC,  True → PR curves / AP
BOXPLOT  = False   # False → mean curve + CI,   True → box plots of AUC/AP
# ─────────────────────────────────────────────────────────────────────────────

# Ablation variant display names and colors
# Ablation: megascale_all (new best) vs other megascale variants + Full (old baseline)
ABLATION_DISPLAY_NAMES = {
    'MutPredPPI_sahni_fragoza_megascale_all':             'MutPred-PPI',
    'MutPredPPI_sahni_fragoza_megascale_freeze_diff':     'Freeze Diff',
    'MutPredPPI_sahni_fragoza_megascale_head':            'Head Only',
    'MutPredPPI_sahni_fragoza_megascale_all_no-gat':      'No GAT',
    'MutPredPPI_sahni_fragoza_megascale_all_no-mut':      'No Mutation Processor',
    'MutPredPPI_sahni_fragoza_megascale_all_wt-emb':      'WT Embedding',
    'MutPredPPI_sahni_fragoza_scratch':                   'No Pretrain',
    'MutPredPPI_sahni_fragoza':                           'Prior Best',
}
# Note: 'Prior Best' is the FoldX-pretrained model from the RECOMB 2024 conference version.
# The paper caption should reference the prior version (bioRxiv / conference proceedings).

ABLATION_COLORS = {
    'MutPred-PPI':              '#1f77b4',  # blue — publication model, same as comparison plots
    'Freeze Diff':              '#4b83c5',
    'Head Only':                '#8c564b',
    'No GAT':                   '#d62728',
    'No Mutation Processor':    '#ff7f0e',
    'WT Embedding':             '#2ca02c',
    'No Pretrain':              '#7f7f7f',
    'Prior Best':               '#aec7e8',  # light blue — distinguishable from MutPred-PPI
}

# Method name mapping (comparison mode) — only methods listed here are plotted

colors = {
    "MutPred-PPI":          "#1f77b4",  # blue — same as blind-test figure
    "GATMutPPI":            "#1f77b4",  # legacy name, same color
    "SWING (Test Pretrain)":"#d62728",
    "SWING (Blind-Test)":   "#ff7f0e",
    "eSIG-Net":             "#9467bd",
    "MutPred2":             "#2ca02c",
    "SAAMBE-3D":            "#8c564b",
    "SAAMBE-3D DN":         "#c4956a",
    "MutPPI":               "#17becf",
    "MutPPI+":              "#bcbd22",
    "DDMutPPI":             "#e377c2",
    "MINT (seq diff)":      "#66c2a5",
    "MINT (site diff)":     "#1b7837",
    "PPLM (seq diff)":      "#4b5563",
    "PPLM (site diff)":     "#762a83",
}

dataset_to_display_name = {
    'sahni':                         'Mendelian',
    'sahni_fragoza':                 'Mendelian and Population',
    'sahni_varchamp1p_cava':         'Mendelian and Benchmark',
    'sahni_fragoza_varchamp1p_cava': 'Mendelian, Population, and Benchmark',
}

# %% Analysis functions
def load_detailed_results(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def extract_ablation_method_and_dataset(filename):
    """Extract method key and dataset from an ablation results filename."""
    basename = os.path.basename(filename).replace('_detailed_results.pkl', '')

    if 'sahni_fragoza_varchamp1p_cava' in basename:
        dataset = 'sahni_fragoza_varchamp1p_cava'
    elif 'sahni_varchamp1p_cava' in basename:
        dataset = 'sahni_varchamp1p_cava'
    elif 'sahni_fragoza' in basename:
        dataset = 'sahni_fragoza'
    elif 'sahni' in basename:
        dataset = 'sahni'
    else:
        dataset = 'unknown'

    method = basename
    return method, dataset


from sklearn.metrics import precision_recall_curve, average_precision_score


def compute_roc_with_variance(detailed_results, prc=False):
    key1, key2 = ('recalls', 'precisions') if prc else ('fprs', 'tprs')
    results = {
        f'class_{c}': {key1: [], key2: [], 'aucs': [], 'ns': []}
        for c in [1, 2, 3]
    }

    for iteration_key in detailed_results['iterations']:
        iteration_data = detailed_results['iterations'][iteration_key]
        class_ns = [0, 0, 0]

        for fold_key in iteration_data['folds']:
            fold_data = iteration_data['folds'][fold_key]

            for class_num in [1, 2, 3]:
                class_key = f'class_{class_num}'
                class_data = fold_data[class_key]

                preds  = np.array(class_data['preds'])
                labels = np.array(class_data['labels'])

                if len(preds) > 0 and len(np.unique(labels)) > 1:
                    if prc:
                        precision, recall, _ = precision_recall_curve(labels, preds)
                        score = average_precision_score(labels, preds)
                        results[class_key][key1].append(recall)
                        results[class_key][key2].append(precision)
                    else:
                        fpr, tpr, _ = roc_curve(labels, preds)
                        score = auc(fpr, tpr)
                        results[class_key][key1].append(fpr)
                        results[class_key][key2].append(tpr)

                    results[class_key]['aucs'].append(score)
                    class_ns[class_num - 1] += len(preds)

        for class_num in [1, 2, 3]:
            results[f'class_{class_num}']['ns'].append(class_ns[class_num - 1])

    return results


def load_baseline_predictions(dataset, prc=False):
    """Load fixed (non-CV) predictor arrays and build per-class ROC inputs."""
    baseline_results = {}
    key1, key2 = ('recalls', 'precisions') if prc else ('fprs', 'tprs')
    auc_lbl = 'AP' if prc else 'AUC'

    labels_file       = os.path.join(WORKING_DIR, f'{dataset}_mutpred2_standalone_labels.npy')
    test_classes_file = os.path.join(WORKING_DIR, f'{dataset}_SAAMBE-3D_test_classes.npy')

    skempi_methods = ['SAAMBE-3D', 'MutPPI', 'MutPPIPlus']  # DDMutPPI excluded: API returns NaN for all variants
    for method in skempi_methods:
        preds_file  = os.path.join(WORKING_DIR, f'{dataset}_{method}_preds.npy')
        binary_file = os.path.join(WORKING_DIR, f'{dataset}_{method}_binary_labels.npy')
        if not (os.path.exists(preds_file) and os.path.exists(labels_file)
                and os.path.exists(test_classes_file)):
            continue
        try:
            preds        = np.load(preds_file)
            labels       = np.load(labels_file)
            test_classes = np.load(test_classes_file)
            bin_labels   = np.load(binary_file) if os.path.exists(binary_file) else None

            method_key = f'{method.replace("-", "_").lower()}_{dataset}'
            baseline_results[method_key] = {}

            for tc in (1, 2, 3):
                mask     = test_classes == tc
                preds_c  = preds[mask]
                labels_c = labels[mask]
                valid    = ~np.isnan(preds_c)
                preds_c  = preds_c[valid]
                labels_c = labels_c[valid]

                if len(preds_c) == 0 or len(np.unique(labels_c)) < 2:
                    baseline_results[method_key][f'class_{tc}'] = {
                        key1: [], key2: [], 'aucs': [], 'ns': [0]
                    }
                    continue

                if prc:
                    precision, recall, _ = precision_recall_curve(labels_c, preds_c)
                    score = average_precision_score(labels_c, preds_c)
                    v1, v2 = recall, precision
                else:
                    fpr, tpr, _ = roc_curve(labels_c, preds_c)
                    score = auc(fpr, tpr)
                    v1, v2 = fpr, tpr

                entry = {key1: [v1], key2: [v2], 'aucs': [score], 'ns': [len(preds_c)]}

                if bin_labels is not None and not prc:
                    bl_c     = bin_labels[mask][valid]
                    valid_bl = bl_c >= 0
                    if np.any(valid_bl) and len(np.unique(bl_c[valid_bl])) >= 2:
                        bl_v  = bl_c[valid_bl]
                        lc_v  = labels_c[valid_bl]
                        tp = np.sum((bl_v == 1) & (lc_v == 1))
                        fn = np.sum((bl_v == 0) & (lc_v == 1))
                        fp = np.sum((bl_v == 1) & (lc_v == 0))
                        tn = np.sum((bl_v == 0) & (lc_v == 0))
                        tpr_pt = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        fpr_pt = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                        entry['binary_pt'] = (fpr_pt, tpr_pt)

                baseline_results[method_key][f'class_{tc}'] = entry
                print(f"  Loaded {method} class {tc}: {auc_lbl}={score:.4f}", flush=True)

        except Exception as e:
            print(f"  Could not load {method} for {dataset}: {e}")

    for method in ['mutpred2_standalone']:
        preds_file = os.path.join(WORKING_DIR, f'{dataset}_{method}_preds.npy')
        if not (os.path.exists(preds_file) and os.path.exists(labels_file)):
            continue
        try:
            preds  = np.load(preds_file)
            labels = np.load(labels_file)
            valid  = ~np.isnan(preds) & (labels >= 0)
            preds  = preds[valid]; labels = labels[valid]

            if prc:
                precision, recall, _ = precision_recall_curve(labels, preds)
                score = average_precision_score(labels, preds)
                v1, v2 = recall, precision
            else:
                fpr, tpr, _ = roc_curve(labels, preds)
                score = auc(fpr, tpr)
                v1, v2 = fpr, tpr

            print(f"  Loaded {method} for {dataset}: {auc_lbl}={score:.4f}")
            method_key = f'{method.replace("-", "_").lower()}_{dataset}'
            baseline_results[method_key] = {
                f'class_{c}': {key1: [v1], key2: [v2], 'aucs': [score], 'ns': [len(preds)]}
                for c in [1, 2, 3]
            }
        except Exception as e:
            print(f"  Could not load {method} for {dataset}: {e}")

    return baseline_results


# %% Plotting functions
def plot_roc_with_confidence(results_dict, dataset_name, save_path=None, prc=False,
                              ablation=False):
    """Plot mean ROC or PR curves with 95% CI shading."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=100)

    key1, key2 = ('recalls', 'precisions') if prc else ('fprs', 'tprs')
    auc_label  = 'AP' if prc else 'AUC'
    xlabel     = 'Recall' if prc else 'False Positive Rate'
    ylabel     = 'Precision' if prc else 'True Positive Rate'
    legend_loc = 'upper right' if prc else 'lower right'

    name_map  = ABLATION_DISPLAY_NAMES if ablation else METHOD_DISPLAY_NAMES
    color_map = ABLATION_COLORS        if ablation else colors

    for class_idx, class_num in enumerate([1, 2, 3]):
        ax = axes[class_idx]
        class_key = f'class_{class_num}'
        legend_items = []

        for method_name, results in results_dict.items():
            if class_key not in results or len(results[class_key]['aucs']) == 0:
                continue

            class_results = results[class_key]
            display_name  = name_map.get(method_name, method_name)
            color         = color_map.get(display_name, '#808080')

            _FIXED_PRED_KEYS = ('mutpred2', 'saambe', 'mutppi', 'ddmutppi')
            is_baseline = (not ablation) and any(
                k in method_name.lower() for k in _FIXED_PRED_KEYS
            )

            mean_x = np.linspace(0, 1, 100)
            ys = []

            for x, y in zip(class_results[key1], class_results[key2]):
                if prc:
                    x, y = x[::-1], y[::-1]
                interp_y = np.interp(mean_x, x, y)
                if not prc:
                    interp_y[0] = 0.0
                ys.append(interp_y)

            ys       = np.array(ys)
            n_curves = 10

            mean_y = np.mean(ys, axis=0)
            if not prc:
                mean_y[-1] = 1.0

            std_y   = np.std(ys, axis=0, ddof=1)
            sem_y   = std_y / np.sqrt(n_curves)
            y_lower = np.clip(mean_y - sem_y, 0, 1)
            y_upper = np.clip(mean_y + sem_y, 0, 1)

            aucs     = np.array(class_results['aucs'])
            mean_auc = np.mean(aucs)
            std_auc  = np.std(aucs, ddof=1)

            if is_baseline:
                n_samples = class_results['ns'][0]
                linestyle = ':' if 'mutpred2' in method_name.lower() else '--'
                lw        = 2.5 if 'mutpred2' in method_name.lower() else 2
                ax.plot(mean_x, mean_y, color=color, lw=lw, alpha=0.8, linestyle=linestyle)
                if not prc and 'binary_pt' in class_results:
                    fpr_pt, tpr_pt = class_results['binary_pt']
                    ax.scatter([fpr_pt], [tpr_pt], marker='*', s=250, color=color,
                               zorder=6, edgecolors='black', linewidths=0.5)
                legend_items.append(
                    (mean_auc, display_name.replace('GATMutPPI', 'MutPred-PPI'),
                     color, None, n_samples, True)
                )
            else:
                ax.plot(mean_x, mean_y, color=color, lw=2, alpha=0.85)
                ax.fill_between(mean_x, y_lower, y_upper, color=color, alpha=0.15)
                legend_items.append(
                    (mean_auc, display_name.replace('GATMutPPI', 'MutPred-PPI'),
                     color, std_auc, n_curves, False)
                )

        if not prc:
            ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)

        legend_items.sort(reverse=True, key=lambda x: x[0])

        handles, labels = [], []
        for item in legend_items:
            mean_auc, name, color, std_auc, n, is_bl = item
            if is_bl:
                handles.append(plt.Line2D(
                    [0], [0], color=color,
                    lw=2.5 if 'mutpred2' in name.lower() else 2,
                    linestyle=':' if 'mutpred2' in name.lower() else '--'
                ))
                labels.append(f'{name} ({auc_label}={mean_auc:.3f}, n={n})')
            else:
                handles.append(plt.Line2D([0], [0], color=color, lw=2))
                labels.append(f'{name} ({auc_label}={mean_auc:.3f}±{std_auc:.3f})')

        _FP = ('mutpred2', 'saambe', 'mutppi', 'ddmutppi')
        for method_name, results in results_dict.items():
            if ablation or not any(k in method_name.lower() for k in _FP):
                n_preds = np.array(results[class_key]['ns'])
                mean_n  = np.mean(n_preds)
                std_n   = np.std(n_preds)
                break

        class_num_to_word = ['One', 'Two', 'Three']
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel(xlabel, fontsize=FONTSIZE_AXIS)
        ax.set_ylabel(ylabel if class_idx == 0 else '', fontsize=FONTSIZE_AXIS)
        ax.set_title(
            f'Class {class_num_to_word[class_num-1]} (n={int(mean_n)}±{int(std_n)})',
            fontsize=TITLE_FONTSIZE
        )
        ax.grid(True, alpha=0.3)
        ax.legend(handles, labels, loc=legend_loc, fontsize=FONTSIZE_LEGEND)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def plot_auc_boxplots(results_dict, dataset_name, save_path=None, prc=False,
                      ablation=False):
    """Box plots of per-fold AUC (or AP) distributions, one panel per class."""
    auc_label = 'AP' if prc else 'AUC'
    name_map  = ABLATION_DISPLAY_NAMES if ablation else METHOD_DISPLAY_NAMES
    color_map = ABLATION_COLORS        if ablation else colors

    class_num_to_word = ['One', 'Two', 'Three']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=100)

    for class_idx, class_num in enumerate([1, 2, 3]):
        ax = axes[class_idx]
        class_key = f'class_{class_num}'

        entries = []
        for method_name, results in results_dict.items():
            if class_key not in results or len(results[class_key]['aucs']) == 0:
                continue

            aucs = np.array(results[class_key]['aucs'])
            if len(aucs) == 0:
                continue

            display_name = name_map.get(method_name, method_name)
            color        = color_map.get(display_name, '#808080')

            _FIXED_PRED_KEYS = ('mutpred2', 'saambe', 'mutppi', 'ddmutppi')
            is_baseline = (not ablation) and any(
                k in method_name.lower() for k in _FIXED_PRED_KEYS
            )

            if is_baseline:
                n_samples = results[class_key]['ns'][0] if results[class_key]['ns'] else 0

            entries.append((np.mean(aucs), display_name, aucs, color, is_baseline))

        entries.sort(reverse=True, key=lambda e: e[0])

        positions   = list(range(len(entries)))
        tick_labels = []

        for pos, (mean_auc, name, aucs, color, is_baseline) in enumerate(entries):
            label_name = name.replace('GATMutPPI', 'MutPred-PPI')

            if is_baseline or len(aucs) == 1:
                ax.hlines(
                    y=pos, xmin=mean_auc - 0.01, xmax=mean_auc + 0.01,
                    colors=color, linewidths=3, linestyles='--', alpha=0.85,
                    zorder=3,
                )
                ax.plot(mean_auc, pos, marker='D', color=color,
                        markersize=7, zorder=4, alpha=0.9)
                tick_labels.append(f'{label_name}\n({auc_label}={mean_auc:.3f})')
            else:
                bp = ax.boxplot(
                    aucs,
                    positions=[pos],
                    vert=False,
                    widths=0.55,
                    patch_artist=True,
                    manage_ticks=False,
                    showfliers=True,
                    flierprops=dict(marker='o', markersize=4,
                                   markerfacecolor=color, markeredgewidth=0.5,
                                   alpha=0.6),
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(color=color, linewidth=1.2, alpha=0.8),
                    capprops=dict(color=color, linewidth=1.5),
                    boxprops=dict(linewidth=0),
                )
                for patch in bp['boxes']:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.75)

                std_auc = np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0
                tick_labels.append(
                    f'{label_name}\n({auc_label}={mean_auc:.3f}±{std_auc:.3f})'
                )

        ax.set_yticks(positions)
        ax.set_yticklabels(tick_labels, fontsize=FONTSIZE_LEGEND)
        ax.set_xlim([0, 1.05])
        ax.set_xlabel(auc_label, fontsize=FONTSIZE_AXIS)
        if class_idx == 0:
            ax.set_ylabel('Method', fontsize=FONTSIZE_AXIS)

        _FP = ('mutpred2', 'saambe', 'mutppi', 'ddmutppi')
        for method_name, results in results_dict.items():
            if class_key in results and len(results[class_key]['ns']) > 0:
                if ablation or not any(k in method_name.lower() for k in _FP):
                    n_arr  = np.array(results[class_key]['ns'])
                    mean_n = np.mean(n_arr)
                    std_n  = np.std(n_arr)
                    break
        else:
            mean_n = std_n = 0

        ax.set_title(
            f'Class {class_num_to_word[class_num - 1]} (n={int(mean_n)}±{int(std_n)})',
            fontsize=TITLE_FONTSIZE,
        )
        ax.grid(True, axis='x', alpha=0.3)
        ax.axvline(x=0.5, color='k', linestyle='--', lw=1.2, alpha=0.4)

    plt.suptitle(
        f'{"AP" if prc else "AUC"} distribution — {dataset_to_display_name.get(dataset_name, dataset_name)}',
        fontsize=TITLE_FONTSIZE + 1,
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def plot_ablation_bars(results_dict, dataset_name, save_path=None, prc=False):
    """Vertical bar chart of mean AUC ± SEM for ablation variants.

    Models on x-axis ordered by C3 performance (MutPred-PPI always first).
    Same model ordering across all three class panels (C1, C2, C3).
    SEM = std / sqrt(10), matching the hardcoded denominator in compute_roc_with_variance.
    """
    auc_label = 'AUC'

    # Determine ordering: MutPred-PPI first, then rest sorted by C3 AUC descending
    def _mean_c3(method_name):
        r = results_dict.get(method_name, {}).get('class_3', {})
        aucs = r.get('aucs', [])
        return np.mean(aucs) if aucs else -1.0

    all_methods = list(results_dict.keys())
    main_method = next(
        (m for m in all_methods if ABLATION_DISPLAY_NAMES.get(m) == 'MutPred-PPI'), None
    )
    others = sorted(
        [m for m in all_methods if m != main_method],
        key=_mean_c3, reverse=True
    )
    ordered_methods = ([main_method] if main_method else []) + others
    ordered_display = [ABLATION_DISPLAY_NAMES.get(m, m) for m in ordered_methods]
    ordered_colors  = [ABLATION_COLORS.get(dn, '#808080') for dn in ordered_display]

    x_pos = np.arange(len(ordered_methods))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=100)

    class_word = {1: 'One', 2: 'Two', 3: 'Three'}

    for class_idx, class_num in enumerate([1, 2, 3]):
        ax = axes[class_idx]
        class_key = f'class_{class_num}'

        for xi, (method_name, display_name, color) in enumerate(
            zip(ordered_methods, ordered_display, ordered_colors)
        ):
            r = results_dict.get(method_name, {}).get(class_key, {})
            aucs = r.get('aucs', [])
            if not aucs:
                continue
            aucs = np.array(aucs)
            mean_auc = float(np.mean(aucs))
            sem_auc  = float(np.std(aucs, ddof=1) / np.sqrt(N_SEM_DIVISOR))
            is_main  = (display_name == 'MutPred-PPI')
            ax.bar(xi, mean_auc, width=0.6,
                   color=color,
                   edgecolor='black' if is_main else color,
                   linewidth=1.5 if is_main else 0.5,
                   alpha=0.9 if is_main else 0.75,
                   zorder=3)
            ax.errorbar(xi, mean_auc, yerr=sem_auc,
                        fmt='none', color='black', capsize=4,
                        linewidth=1.5, zorder=4)
            # Value label above error bar
            ax.text(xi, mean_auc + sem_auc + 0.005, f'{mean_auc:.3f}',
                    ha='center', va='bottom', fontsize=7.5,
                    fontweight='bold' if is_main else 'normal', color='black')

        # Sample size from MutPred-PPI (most complete)
        ref_ns = (results_dict.get(main_method or ordered_methods[0], {})
                  .get(class_key, {}).get('ns', []))
        n_label = f'n={int(np.mean(ref_ns))}' if ref_ns else ''
        ax.set_title(
            f'Class {class_word[class_num]} ({n_label})',
            fontsize=TITLE_FONTSIZE,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ordered_display, rotation=45, ha='right',
                           fontsize=FONTSIZE_LEGEND)
        ax.set_ylim([0.5, 1.02])  # extra headroom for value labels above error bars
        ax.set_ylabel(auc_label if class_idx == 0 else '', fontsize=FONTSIZE_AXIS)
        ax.axhline(y=0.5, color='k', linestyle='--', lw=1.2, alpha=0.4)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_xlim(-0.6, len(ordered_methods) - 0.4)

    plt.suptitle(
        f'Ablation — {dataset_to_display_name.get(dataset_name, dataset_name)}',
        fontsize=TITLE_FONTSIZE + 1,
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        # Always save with 'ablation_bar' prefix regardless of BOXPLOT flag
        bar_path = save_path.replace('ablation_boxplot_', 'ablation_bar_')
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {bar_path}")
    plt.show()
    return fig


def _dispatch_plot(results_dict, dataset_name, save_path, prc, ablation):
    if ablation and BOXPLOT:
        plot_ablation_bars(results_dict, dataset_name, save_path, prc=prc)
    elif BOXPLOT:
        plot_auc_boxplots(results_dict, dataset_name, save_path, prc=prc, ablation=ablation)
    else:
        plot_roc_with_confidence(results_dict, dataset_name, save_path, prc=prc, ablation=ablation)


# %% Main entry points
def main():
    if ABLATION:
        return main_ablation()
    else:
        return main_comparison()


def main_comparison():
    """Compare methods across datasets.

    Only files whose extracted method key is in METHOD_DISPLAY_NAMES are
    included — ablation variants and unrecognized files are silently skipped.
    """
    detailed_files = glob.glob(os.path.join(WORKING_DIR, "*_detailed_results.pkl"))
    detailed_files = [f for f in detailed_files
                      if 'mutpred2' not in f.lower() and 'saambe' not in f.lower()]

    print(f"Found {len(detailed_files)} detailed results files")

    datasets_results = {}

    for filepath in detailed_files:
        print(f"\nProcessing: {os.path.basename(filepath)}")
        method, dataset = extract_method_and_dataset(filepath)

        if method is None or method not in METHOD_DISPLAY_NAMES:
            print(f"  Skipping: not a recognized comparison method (method={method!r})")
            continue

        try:
            detailed_results = load_detailed_results(filepath)
            roc_results = compute_roc_with_variance(detailed_results, prc=PRC)

            if dataset not in datasets_results:
                datasets_results[dataset] = {}
            datasets_results[dataset][method] = roc_results

            for class_num in [1, 2, 3]:
                class_key = f'class_{class_num}'
                n_curves = len(roc_results[class_key]['aucs'])
                if n_curves > 0:
                    mean_auc = np.mean(roc_results[class_key]['aucs'])
                    std_auc  = np.std(roc_results[class_key]['aucs'])
                    print(f"  Class {class_num}: AUC={mean_auc:.4f}±{std_auc:.4f} (n={n_curves})")

        except Exception as e:
            print(f"  Error processing {filepath}: {e}")

    for dataset in datasets_results.keys():
        print(f"\nLoading baseline predictors for {dataset}")
        baseline_results = load_baseline_predictions(dataset, prc=PRC)
        datasets_results[dataset].update(baseline_results)

    _ds = ['sahni', 'sahni_fragoza', 'sahni_fragoza_varchamp1p_cava', 'sahni_fragoza_varchamp2026',
           'sahni_fragoza_varchamp_full', 'sahni_fragoza_varchamp_pooled',
           'sahni_fragoza_varchamp_full_pooled']
    # Was an in-place update of the module-level dict; now a local extended copy,
    # so sharing the mapping cannot leak baseline keys into other consumers.
    globals()['METHOD_DISPLAY_NAMES'] = with_baseline_variants(_ds)

    if SAVE_PLOTS:
        os.makedirs(os.path.join(WORKING_DIR, SAVE_DIR), exist_ok=True)

    for dataset, methods_results in datasets_results.items():
        print(f"\n{'='*60}")
        print(f"Creating plot for dataset: {dataset}")
        print(f"Methods: {list(methods_results.keys())}")

        if SAVE_PLOTS:
            suffix = 'boxplot' if BOXPLOT else 'roc'
            save_path = os.path.join(WORKING_DIR, SAVE_DIR,
                                     f"{suffix}_{dataset}_with_variance.png")
        else:
            save_path = None

        _dispatch_plot(methods_results, dataset, save_path, prc=PRC, ablation=False)

    return datasets_results


def main_ablation():
    """Compare MutPredPPI ablation variants.

    Only files whose extracted method key is in ABLATION_DISPLAY_NAMES are
    included — comparison methods and unregistered variants are skipped.
    """
    ablation_files = glob.glob(os.path.join(WORKING_DIR, "MutPredPPI_*_detailed_results.pkl"))

    print(f"Found {len(ablation_files)} ablation results files")

    datasets_results = {}

    for filepath in ablation_files:
        print(f"\nProcessing: {os.path.basename(filepath)}")
        method, dataset = extract_ablation_method_and_dataset(filepath)

        if method not in ABLATION_DISPLAY_NAMES:
            print(f"  Skipping: not in ABLATION_DISPLAY_NAMES (method={method!r})")
            continue

        try:
            detailed_results = load_detailed_results(filepath)
            roc_results = compute_roc_with_variance(detailed_results, prc=PRC)

            if dataset not in datasets_results:
                datasets_results[dataset] = {}
            datasets_results[dataset][method] = roc_results

            for class_num in [1, 2, 3]:
                class_key = f'class_{class_num}'
                n_curves = len(roc_results[class_key]['aucs'])
                if n_curves > 0:
                    mean_auc = np.mean(roc_results[class_key]['aucs'])
                    std_auc  = np.std(roc_results[class_key]['aucs'])
                    print(f"  Class {class_num}: AUC={mean_auc:.4f}±{std_auc:.4f} (n={n_curves})")

        except Exception as e:
            print(f"  Error processing {filepath}: {e}")

    if SAVE_PLOTS:
        os.makedirs(os.path.join(WORKING_DIR, SAVE_DIR), exist_ok=True)

    for dataset, methods_results in datasets_results.items():
        print(f"\n{'='*60}")
        print(f"Creating ablation plot for dataset: {dataset}")
        print(f"Variants: {list(methods_results.keys())}")

        if SAVE_PLOTS:
            suffix = 'ablation_boxplot' if BOXPLOT else 'ablation'
            save_path = os.path.join(WORKING_DIR, SAVE_DIR,
                                     f"{suffix}_{dataset}_with_variance.png")
        else:
            save_path = None

        _dispatch_plot(methods_results, dataset, save_path, prc=PRC, ablation=True)

    return datasets_results


# %% Run
if __name__ == '__main__':
    results = main()
