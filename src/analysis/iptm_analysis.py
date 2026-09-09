#!/usr/bin/env python
"""AlphaFold 3 ipTM / pTM comparison against MutPred-PPI performance.

Split out of `roc_plots.py`, where it was the second half of a 1,949-line module
and — because that file carried **two** `if __name__ == "__main__"` blocks —
running `python roc_plots.py` silently executed the ROC figure generation *and*
this analysis. They are separate concerns with separate outputs.

The coupling was one name: `WORKING_DIR`. Everything else here is self-contained.

Writes `iptm_{dataset}_gcv_splits.pkl`, which `biclass_sf_gcv.py` consumes.

Usage:
    conda run -n ppi python src/analysis/iptm_analysis.py
"""
from __future__ import annotations

import glob
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
from paths import CV_DIR, GCV_RESULTS_DIR  # noqa: E402

WORKING_DIR = str(GCV_RESULTS_DIR)
CV_REF = str(CV_DIR)


# %% [markdown]
# ### ipTM / pTM Comparison

# %% ipTM Configuration
PLOT_MODE   = 'spearman_box'   # 'curve' | 'boxplot' | 'scatter' | 'spearman_box'
SCORE_TYPE  = 'iptm'           # 'iptm' | 'ptm'
IPTM_THRESH = 0.6
PRC         = False
SAVE_PLOTS  = True
SAVE_DIR    = "roc_plots_with_variance"

FOR_SLIDES      = False
TITLE_FONTSIZE  = 20 if FOR_SLIDES else 14
FONTSIZE_LEGEND = 11 if FOR_SLIDES else 9
FONTSIZE_AXIS   = 16 if FOR_SLIDES else 12

TARGET_DATASET      = 'sahni_fragoza'
METHOD_DISPLAY_NAME = 'MutPred-PPI'

HIGH_COLOR = '#2E7D32'
LOW_COLOR  = '#C62828'

dataset_to_display_name = {
    'sahni':                         'Mendelian',
    'sahni_fragoza':                 'Mendelian and Population',
    'sahni_varchamp1p_cava':         'Mendelian and Benchmark',
    'sahni_fragoza_varchamp1p_cava': 'Mendelian, Population, and Benchmark',
}

# %% ipTM helper functions
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _score(labels, preds, prc=False):
    if prc:
        precision, recall, _ = precision_recall_curve(labels, preds)
        return recall[::-1], precision[::-1], average_precision_score(labels, preds)
    fpr, tpr, _ = roc_curve(labels, preds)
    return fpr, tpr, auc(fpr, tpr)


def build_complex_auc_table(detailed_results, iptm_detailed_results,
                             score_key='iptms', prc=PRC):
    """Pool predictions per complex across all fold-iterations, compute AUC."""
    pool = {}
    for c in [1, 2, 3]:
        pool[f'class_{c}'] = {}

    for iter_key in detailed_results['iterations']:
        iter_data      = detailed_results['iterations'][iter_key]
        iptm_iter_data = iptm_detailed_results['iterations'][iter_key]

        for fold_key in iter_data['folds']:
            fold_data      = iter_data['folds'][fold_key]
            iptm_fold_data = iptm_iter_data['folds'][fold_key]

            for c in [1, 2, 3]:
                ck = f'class_{c}'
                if ck not in fold_data or ck not in iptm_fold_data:
                    continue

                preds      = np.array(fold_data[ck].get('preds',       []))
                labels     = np.array(fold_data[ck].get('labels',      []))
                scores_arr = np.array(iptm_fold_data[ck].get(score_key, []))
                cids       = iptm_fold_data[ck].get('complex_ids',      [])

                if len(preds) == 0 or len(cids) == 0:
                    continue

                min_len    = min(len(preds), len(labels), len(scores_arr), len(cids))
                preds      = preds[:min_len]
                labels     = labels[:min_len]
                scores_arr = scores_arr[:min_len]
                cids       = cids[:min_len]

                for pred, label, score_val, cid in zip(preds, labels, scores_arr, cids):
                    if np.isnan(score_val):
                        continue
                    if cid not in pool[ck]:
                        pool[ck][cid] = {'score': score_val, 'preds': [], 'labels': []}
                    pool[ck][cid]['preds'].append(pred)
                    pool[ck][cid]['labels'].append(label)

    result = {}
    for c in [1, 2, 3]:
        ck = f'class_{c}'
        result[ck] = {}
        for cid, d in pool[ck].items():
            lbl = np.array(d['labels'])
            prd = np.array(d['preds'])
            if len(np.unique(lbl)) < 2:
                continue
            _, _, sc = _score(lbl, prd, prc)
            result[ck][cid] = {
                'score':    d['score'],
                'n_points': len(prd),
                'auc':      sc,
            }

    return result


def build_binned_results(detailed_results, iptm_detailed_results,
                          score_key='iptms', threshold=IPTM_THRESH, prc=PRC):
    """Split data into high/low bins by score_key, compute ROC per fold-iteration."""
    key1 = 'recalls' if prc else 'fprs'
    key2 = 'precisions' if prc else 'tprs'
    template = lambda: {ck: {key1: [], key2: [], 'aucs': [], 'n_points': []}
                        for ck in ['class_1', 'class_2', 'class_3']}
    results = {'high': template(), 'low': template()}

    for iter_key in detailed_results['iterations']:
        iter_data      = detailed_results['iterations'][iter_key]
        iptm_iter_data = iptm_detailed_results['iterations'][iter_key]

        iter_n = {'high': {f'class_{c}': 0 for c in [1, 2, 3]},
                  'low':  {f'class_{c}': 0 for c in [1, 2, 3]}}

        for fold_key in iter_data['folds']:
            fold_data      = iter_data['folds'][fold_key]
            iptm_fold_data = iptm_iter_data['folds'][fold_key]

            for c in [1, 2, 3]:
                ck = f'class_{c}'
                if ck not in fold_data or ck not in iptm_fold_data:
                    continue

                preds      = np.array(fold_data[ck].get('preds',       []))
                labels     = np.array(fold_data[ck].get('labels',      []))
                scores_arr = np.array(iptm_fold_data[ck].get(score_key, []))

                if len(preds) == 0:
                    continue

                min_len = min(len(preds), len(labels), len(scores_arr))
                preds, labels, scores_arr = (preds[:min_len], labels[:min_len],
                                             scores_arr[:min_len])

                valid = ~np.isnan(scores_arr)
                preds, labels, scores_arr = preds[valid], labels[valid], scores_arr[valid]
                if len(preds) == 0:
                    continue

                for bin_name, mask in [('high', scores_arr >  threshold),
                                       ('low',  scores_arr <= threshold)]:
                    p_b, l_b = preds[mask], labels[mask]
                    iter_n[bin_name][ck] += int(np.sum(mask))
                    if len(p_b) == 0 or len(np.unique(l_b)) < 2:
                        continue
                    x, y, sc = _score(l_b, p_b, prc)
                    results[bin_name][ck][key1].append(x)
                    results[bin_name][ck][key2].append(y)
                    results[bin_name][ck]['aucs'].append(sc)

        for bin_name in ['high', 'low']:
            for c in [1, 2, 3]:
                ck = f'class_{c}'
                if iter_n[bin_name][ck] > 0:
                    results[bin_name][ck]['n_points'].append(iter_n[bin_name][ck])

    return results


# %% ipTM plot functions
def plot_iptm_curves(binned, score_label, dataset_name, save_path=None, prc=PRC):
    key1 = 'recalls' if prc else 'fprs'
    key2 = 'precisions' if prc else 'tprs'
    auc_label  = 'AP' if prc else 'AUC'
    xlabel     = 'Recall' if prc else 'False Positive Rate'
    ylabel     = 'Precision' if prc else 'True Positive Rate'
    legend_loc = 'upper right' if prc else 'lower right'
    words      = ['One', 'Two', 'Three']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=100)

    for ci, c in enumerate([1, 2, 3]):
        ax = axes[ci]
        ck = f'class_{c}'

        for bin_name, color, lbl in [
            ('high', HIGH_COLOR, f'{score_label} > {IPTM_THRESH}'),
            ('low',  LOW_COLOR,  f'{score_label} ≤ {IPTM_THRESH}'),
        ]:
            br = binned[bin_name][ck]
            if not br['aucs']:
                continue

            mean_x = np.linspace(0, 1, 100)
            ys = []
            for x, y in zip(br[key1], br[key2]):
                iy = np.interp(mean_x, x[::-1] if prc else x, y[::-1] if prc else y)
                if not prc:
                    iy[0] = 0.0
                ys.append(iy)
            ys     = np.array(ys)
            mean_y = np.mean(ys, axis=0)
            if not prc:
                mean_y[-1] = 1.0
            sem_y  = np.std(ys, axis=0, ddof=1) / np.sqrt(len(ys))
            y_lo   = np.clip(mean_y - sem_y, 0, 1)
            y_hi   = np.clip(mean_y + sem_y, 0, 1)

            mean_auc = np.mean(br['aucs'])
            std_auc  = np.std(br['aucs'], ddof=1)
            mean_n   = int(np.mean(br['n_points']))

            ax.plot(mean_x, mean_y, color=color, lw=2,
                    label=f'{lbl} ({auc_label}={mean_auc:.3f}±{std_auc:.3f}, n={mean_n})')
            ax.fill_between(mean_x, y_lo, y_hi, color=color, alpha=0.15)

        if not prc:
            ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)

        all_n = sum(int(np.mean(binned[b][ck]['n_points']))
                    for b in ['high', 'low'] if binned[b][ck]['n_points'])
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        ax.set_xlabel(xlabel, fontsize=FONTSIZE_AXIS)
        ax.set_ylabel(ylabel if ci == 0 else '', fontsize=FONTSIZE_AXIS)
        ax.set_title(f'Class {words[c-1]} (n={all_n} variants)', fontsize=TITLE_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(loc=legend_loc, fontsize=FONTSIZE_LEGEND)

    plt.suptitle(f'{METHOD_DISPLAY_NAME} – {score_label} Bins – '
                 f'{dataset_to_display_name.get(dataset_name, dataset_name)}',
                 fontsize=TITLE_FONTSIZE, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


VIOLIN_MODE = 'both'   # 'box' | 'violin' | 'both'

def _sig_label(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

def _paired_aucs(binned, ck):
    hi = np.array(binned['high'][ck]['aucs'])
    lo = np.array(binned['low'][ck]['aucs'])
    n  = min(len(hi), len(lo))
    return hi[:n], lo[:n]

def _draw_group(ax, data_sets, positions, bin_colors, mode):
    if mode in ('violin', 'both'):
        valid = [(d, p, c) for d, p, c in zip(data_sets, positions, bin_colors) if len(d)]
        if valid:
            vp = ax.violinplot([v[0] for v in valid],
                               positions=[v[1] for v in valid],
                               widths=0.5, showmedians=False, showextrema=False)
            for body, (_, _, col) in zip(vp['bodies'], valid):
                body.set_facecolor(col); body.set_alpha(0.35)
                body.set_edgecolor(col); body.set_linewidth(1.2)

    if mode in ('box', 'both'):
        w = 0.18 if mode == 'both' else 0.45
        valid = [(d, p, c) for d, p, c in zip(data_sets, positions, bin_colors) if len(d)]
        if valid:
            bp = ax.boxplot([v[0] for v in valid],
                            positions=[v[1] for v in valid],
                            widths=w, patch_artist=True,
                            medianprops=dict(color='white', linewidth=2.5),
                            whiskerprops=dict(linewidth=1.5),
                            capprops=dict(linewidth=1.5),
                            flierprops=dict(marker='o', markersize=3,
                                           alpha=0.0 if mode == 'both' else 0.5))
            for patch, (_, _, col) in zip(bp['boxes'], valid):
                patch.set_facecolor(col); patch.set_alpha(0.85)
            if mode == 'both':
                for flier in bp['fliers']:
                    flier.set_visible(False)

def _sig_bracket(ax, pos_l, pos_r, data_l, data_r, y_offset=0.0):
    from scipy.stats import wilcoxon
    stat_str = ''
    if len(data_l) >= 10 and len(data_r) >= 10:
        n = min(len(data_l), len(data_r))
        dl, dr = np.array(data_l[:n]), np.array(data_r[:n])
        if not np.all(dl == dr):
            try:
                _, p = wilcoxon(dl, dr, alternative='two-sided')
                stat_str = f'p={p:.3g} {_sig_label(p)}'
            except Exception:
                stat_str = 'test failed'
    if stat_str and len(data_l) and len(data_r):
        y_top = max(np.max(data_l), np.max(data_r)) + 0.05 + y_offset
        ax.plot([pos_l, pos_l, pos_r, pos_r],
                [y_top, y_top + 0.02, y_top + 0.02, y_top], lw=1.5, color='#333')
        ax.text((pos_l + pos_r) / 2, y_top + 0.025, stat_str,
                ha='center', va='bottom', fontsize=9, color='#333')
        ax.text((pos_l + pos_r) / 2, y_top + 0.065,
                '⚠ bins unbalanced in size',
                ha='center', va='bottom', fontsize=7, color='#888', style='italic')
        return y_top + 0.10
    return y_offset

def plot_iptm_boxplots(binned_iptm, binned_ptm, dataset_name, save_path=None, prc=PRC):
    auc_label = 'AP' if prc else 'AUC'
    words     = ['One', 'Two', 'Three']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=100)

    iptm_pos   = [1.0, 2.0]
    ptm_pos    = [3.5, 4.5]
    bin_colors = [HIGH_COLOR, LOW_COLOR]

    for ci, c in enumerate([1, 2, 3]):
        ax  = axes[ci]
        ck  = f'class_{c}'

        ih   = np.array(binned_iptm['high'][ck]['aucs'])
        il   = np.array(binned_iptm['low'][ck]['aucs'])
        ih_n = int(np.mean(binned_iptm['high'][ck]['n_points'])) if len(ih) else 0
        il_n = int(np.mean(binned_iptm['low'][ck]['n_points']))  if len(il) else 0

        ph   = np.array(binned_ptm['high'][ck]['aucs'])
        pl   = np.array(binned_ptm['low'][ck]['aucs'])
        ph_n = int(np.mean(binned_ptm['high'][ck]['n_points'])) if len(ph) else 0
        pl_n = int(np.mean(binned_ptm['low'][ck]['n_points']))  if len(pl) else 0

        _draw_group(ax, [ih, il], iptm_pos, bin_colors, VIOLIN_MODE)
        _draw_group(ax, [ph, pl], ptm_pos,  bin_colors, VIOLIN_MODE)

        _sig_bracket(ax, iptm_pos[0], iptm_pos[1], ih, il)
        _sig_bracket(ax, ptm_pos[0],  ptm_pos[1],  ph, pl)

        for pos, data, color in zip(iptm_pos + ptm_pos,
                                    [ih, il, ph, pl],
                                    bin_colors + bin_colors):
            if len(data):
                med = np.median(data)
                ax.text(pos, med + 0.02, f'{med:.3f}',
                        ha='center', va='bottom', fontsize=8,
                        color=color, fontweight='bold')

        ax.set_xticks(iptm_pos + ptm_pos)
        ax.set_xticklabels([
            f'ipTM > {IPTM_THRESH}\n(n={ih_n})',
            f'ipTM ≤ {IPTM_THRESH}\n(n={il_n})',
            f'pTM > {IPTM_THRESH}\n(n={ph_n})',
            f'pTM ≤ {IPTM_THRESH}\n(n={pl_n})',
        ], fontsize=FONTSIZE_LEGEND - 1)

        ax.text(1.5, -0.14, 'ipTM', ha='center', va='top',
                transform=ax.get_xaxis_transform(),
                fontsize=FONTSIZE_LEGEND, fontweight='bold', color='#444')
        ax.text(4.0, -0.14, 'pTM', ha='center', va='top',
                transform=ax.get_xaxis_transform(),
                fontsize=FONTSIZE_LEGEND, fontweight='bold', color='#444')

        ax.axvline(2.75, color='#ccc', lw=1.2, ls='--', zorder=0)

        ax.set_ylabel(auc_label if ci == 0 else '', fontsize=FONTSIZE_AXIS)
        ax.set_title(f'Class {words[c-1]}', fontsize=TITLE_FONTSIZE)
        ax.set_ylim([0, 1.18])
        ax.set_xlim([0.3, 5.2])
        ax.grid(True, axis='y', alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=HIGH_COLOR, alpha=0.75, label=f'High (> {IPTM_THRESH})'),
        Patch(facecolor=LOW_COLOR,  alpha=0.75, label=f'Low (≤ {IPTM_THRESH})'),
    ]
    fig.legend(handles=legend_elements, loc='upper center',
               ncol=2, fontsize=FONTSIZE_LEGEND,
               bbox_to_anchor=(0.5, 1.04), frameon=False)

    plt.suptitle(f'{METHOD_DISPLAY_NAME} – {auc_label} by ipTM / pTM Bin – '
                 f'{dataset_to_display_name.get(dataset_name, dataset_name)}',
                 fontsize=TITLE_FONTSIZE, y=1.09)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def _spearman_ci(xs, ys, alpha=0.05):
    from scipy.stats import spearmanr
    from scipy.special import ndtri
    n = len(xs)
    rho, p = spearmanr(xs, ys)
    if n < 4:
        return rho, p, np.nan, np.nan
    z      = np.arctanh(rho)
    se     = 1.0 / np.sqrt(n - 3)
    z_crit = ndtri(1 - alpha / 2)
    ci_lo  = np.tanh(z - z_crit * se)
    ci_hi  = np.tanh(z + z_crit * se)
    return rho, p, ci_lo, ci_hi

def plot_complex_scatter(complex_auc_table, score_label, dataset_name,
                         save_path=None, prc=PRC):
    from scipy.stats import spearmanr
    auc_label = 'AP' if prc else 'AUC'
    words     = ['One', 'Two', 'Three']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=100)

    for ci, c in enumerate([1, 2, 3]):
        ax = axes[ci]
        ck = f'class_{c}'

        if not complex_auc_table.get(ck):
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, alpha=0.5)
            ax.set_title(f'Class {words[c-1]}', fontsize=TITLE_FONTSIZE)
            continue

        entries = list(complex_auc_table[ck].values())
        xs      = np.array([e['score']    for e in entries])
        ys      = np.array([e['auc']      for e in entries])
        ns      = np.array([e['n_points'] for e in entries])

        high_mask = xs >  IPTM_THRESH
        low_mask  = xs <= IPTM_THRESH

        ss = np.clip(ns * 1.5, 8, 80)
        scatter_kwargs = dict(alpha=0.65, linewidths=0.4, edgecolors='white')

        if np.any(high_mask):
            ax.scatter(xs[high_mask], ys[high_mask], c=HIGH_COLOR,
                       s=ss[high_mask],
                       label=f'{score_label} > {IPTM_THRESH} (n={np.sum(high_mask)})',
                       **scatter_kwargs, zorder=3)
        if np.any(low_mask):
            ax.scatter(xs[low_mask], ys[low_mask], c=LOW_COLOR,
                       s=ss[low_mask],
                       label=f'{score_label} ≤ {IPTM_THRESH} (n={np.sum(low_mask)})',
                       **scatter_kwargs, zorder=3)

        if len(xs) >= 5:
            rho, p_sp, ci_lo, ci_hi = _spearman_ci(xs, ys)
            stars = _sig_label(p_sp)

            m, b  = np.polyfit(xs, ys, 1)
            x_fit = np.linspace(xs.min(), xs.max(), 200)
            ax.plot(x_fit, m * x_fit + b, color='#333', lw=2, ls='--', alpha=0.7,
                    zorder=4)

            ci_str  = f'[{ci_lo:+.2f}, {ci_hi:+.2f}]'
            p_str   = f'p={p_sp:.2e}' if p_sp < 0.001 else f'p={p_sp:.3f}'
            ann_txt = (f'Spearman r = {rho:+.3f}\n'
                       f'95% CI {ci_str}\n'
                       f'{p_str}  {stars}')
            ax.text(0.03, 0.97, ann_txt,
                    transform=ax.transAxes, fontsize=9, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='#ccc'),
                    zorder=5)

            print(f"  Class {c} Spearman: r={rho:+.4f} 95%CI [{ci_lo:.4f},{ci_hi:.4f}] "
                  f"{p_str} {stars} (n={len(xs)} complexes)")

        ax.axvline(IPTM_THRESH, color='#999', lw=1.2, ls=':', alpha=0.7)
        ax.text(0.03, 0.03, 'Dot size ∝ n variants',
                transform=ax.transAxes, fontsize=8, va='bottom', alpha=0.6)

        total_complexes = len(entries)
        total_variants  = int(np.sum(ns))

        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        ax.set_xlabel(f'{score_label} (per complex)', fontsize=FONTSIZE_AXIS)
        ax.set_ylabel(auc_label if ci == 0 else '', fontsize=FONTSIZE_AXIS)
        ax.set_title(
            f'Class {words[c-1]} ({total_complexes} complexes, {total_variants} variants)',
            fontsize=TITLE_FONTSIZE
        )
        ax.grid(True, alpha=0.25)
        ax.legend(loc='lower right', fontsize=FONTSIZE_LEGEND)

    plt.suptitle(
        f'{METHOD_DISPLAY_NAME} – {auc_label} vs. {score_label} per Complex – '
        f'{dataset_to_display_name.get(dataset_name, dataset_name)}',
        fontsize=TITLE_FONTSIZE, y=1.02
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def build_fold_spearman_table(detailed_results, iptm_detailed_results,
                               complex_auc_iptm, complex_auc_ptm, prc=PRC):
    """Compute Spearman r(score, AUC) for each fold x iteration."""
    from scipy.stats import spearmanr

    result = {f'class_{c}': {'iptm': [], 'ptm': []} for c in [1, 2, 3]}

    for iter_key in iptm_detailed_results['iterations']:
        iptm_iter_data = iptm_detailed_results['iterations'][iter_key]

        for fold_key in iptm_iter_data['folds']:
            iptm_fold_data = iptm_iter_data['folds'][fold_key]

            for c in [1, 2, 3]:
                ck = f'class_{c}'
                if ck not in iptm_fold_data:
                    continue

                cids   = iptm_fold_data[ck].get('complex_ids', [])
                iptms  = np.array(iptm_fold_data[ck].get('iptms', []))
                ptms   = np.array(iptm_fold_data[ck].get('ptms',  []))

                if len(cids) == 0:
                    continue

                min_len = min(len(cids), len(iptms), len(ptms))
                cids  = cids[:min_len]
                iptms = iptms[:min_len]
                ptms  = ptms[:min_len]

                fold_iptm, fold_ptm, fold_auc = [], [], []
                seen = set()
                for cid, iv, pv in zip(cids, iptms, ptms):
                    if cid in seen:
                        continue
                    seen.add(cid)
                    if np.isnan(iv) or np.isnan(pv):
                        continue
                    entry = complex_auc_iptm.get(ck, {}).get(cid)
                    if entry is None:
                        continue
                    fold_iptm.append(iv)
                    fold_ptm.append(pv)
                    fold_auc.append(entry['auc'])

                if len(fold_auc) < 5:
                    result[ck]['iptm'].append(np.nan)
                    result[ck]['ptm'].append(np.nan)
                    continue

                fold_iptm = np.array(fold_iptm)
                fold_ptm  = np.array(fold_ptm)
                fold_auc  = np.array(fold_auc)

                r_iptm, _ = spearmanr(fold_iptm, fold_auc)
                r_ptm,  _ = spearmanr(fold_ptm,  fold_auc)

                result[ck]['iptm'].append(r_iptm)
                result[ck]['ptm'].append(r_ptm)

    for c in [1, 2, 3]:
        ck = f'class_{c}'
        ri = np.array(result[ck]['iptm'])
        rp = np.array(result[ck]['ptm'])
        valid = ~(np.isnan(ri) | np.isnan(rp))
        result[ck]['iptm'] = ri[valid]
        result[ck]['ptm']  = rp[valid]

    return result


def plot_spearman_boxplot(spearman_table, dataset_name, save_path=None):
    """Side-by-side violin+box of Spearman r distributions for ipTM and pTM."""
    IPTM_COL = '#1f77b4'
    PTM_COL  = '#ff7f0e'

    words = ['One', 'Two', 'Three']
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), dpi=100)

    for ci, c in enumerate([1, 2, 3]):
        ax = axes[ci]
        ck = f'class_{c}'

        ri = spearman_table[ck]['iptm']
        rp = spearman_table[ck]['ptm']

        positions   = [1, 2]
        data_sets   = [ri, rp]
        _colors     = [IPTM_COL, PTM_COL]
        score_names = ['ipTM', 'pTM']
        labels_x    = [
            'ipTM\n(n=' + str(len(ri)) + ')',
            'pTM\n(n='  + str(len(rp)) + ')',
        ]

        if VIOLIN_MODE in ('violin', 'both'):
            valid = [(d, p, col) for d, p, col in zip(data_sets, positions, _colors) if len(d)]
            if valid:
                vp = ax.violinplot([v[0] for v in valid],
                                   positions=[v[1] for v in valid],
                                   widths=0.5, showmedians=False, showextrema=False)
                for body, (_, _, col) in zip(vp['bodies'], valid):
                    body.set_facecolor(col); body.set_alpha(0.35)
                    body.set_edgecolor(col); body.set_linewidth(1.2)

        if VIOLIN_MODE in ('box', 'both'):
            w = 0.18 if VIOLIN_MODE == 'both' else 0.45
            valid = [(d, p, col) for d, p, col in zip(data_sets, positions, _colors) if len(d)]
            if valid:
                bp = ax.boxplot(
                    [v[0] for v in valid],
                    positions=[v[1] for v in valid],
                    widths=w, patch_artist=True,
                    medianprops=dict(color='white', linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markersize=3,
                                   alpha=0.0 if VIOLIN_MODE == 'both' else 0.5),
                )
                for patch, (_, _, col) in zip(bp['boxes'], valid):
                    patch.set_facecolor(col); patch.set_alpha(0.85)
                if VIOLIN_MODE == 'both':
                    for flier in bp['fliers']:
                        flier.set_visible(False)

        for pos, data, col, sname in zip(positions, data_sets, _colors, score_names):
            if len(data) < 5:
                continue
            med   = np.median(data)
            lo_ci = np.percentile(data, 2.5)
            hi_ci = np.percentile(data, 97.5)
            sig   = lo_ci > 0 or hi_ci < 0

            med_str = ('+' if med >= 0 else '') + f'{med:.3f}'
            ax.text(pos + (0.12 if VIOLIN_MODE == 'box' else 0.28),
                    med, med_str,
                    ha='left', va='center', fontsize=8,
                    color=col, fontweight='bold')

            lo_str  = ('+' if lo_ci >= 0 else '') + f'{lo_ci:.3f}'
            hi_str  = ('+' if hi_ci >= 0 else '') + f'{hi_ci:.3f}'
            sig_str = '*' if sig else 'ns'
            ann     = '[' + lo_str + ', ' + hi_str + ']  ' + sig_str
            y_ann   = float(np.max(data)) + 0.03
            ax.text(pos, y_ann, ann,
                    ha='center', va='bottom', fontsize=8,
                    color=col, fontweight='bold' if sig else 'normal',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))

            print('  Class ' + str(c) + ' ' + sname + ': median r=' + med_str +
                  ' 95% CI [' + lo_str + ', ' + hi_str + '] -> ' +
                  ('significant' if sig else 'ns'))

        ax.axhline(0, color='#aaa', lw=1.2, ls='--', zorder=0)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels_x, fontsize=FONTSIZE_LEGEND)
        ax.set_ylabel('Spearman r' if ci == 0 else '', fontsize=FONTSIZE_AXIS)
        ax.set_title('Class ' + words[c-1], fontsize=TITLE_FONTSIZE)
        ax.set_xlim([0.4, 2.6])
        all_vals = np.concatenate([ri, rp]) if (len(ri) and len(rp)) else (ri if len(ri) else rp)
        if len(all_vals):
            ylo = min(float(np.min(all_vals)) - 0.05, -0.1)
            yhi = max(float(np.max(all_vals)) + 0.22,  0.3)
            ax.set_ylim([ylo, yhi])
        ax.grid(True, axis='y', alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=IPTM_COL, alpha=0.75, label='ipTM'),
        Patch(facecolor=PTM_COL,  alpha=0.75, label='pTM'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
               fontsize=FONTSIZE_LEGEND, bbox_to_anchor=(0.5, 1.04), frameon=False)

    dname = dataset_to_display_name.get(dataset_name, dataset_name)
    plt.suptitle(
        METHOD_DISPLAY_NAME + ' – Spearman r per fold – ' + dname,
        fontsize=TITLE_FONTSIZE, y=1.09,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print('Saved: ' + save_path)
    plt.show()
    return fig


# %% ipTM main
def main_iptm_analysis():
    """
    ipTM / pTM analysis for MutPred-PPI on TARGET_DATASET.

    PLOT_MODE  : 'curve' | 'boxplot' | 'scatter' | 'spearman_box'
    SCORE_TYPE : 'iptm'  | 'ptm'
    """
    score_key   = 'iptms' if SCORE_TYPE == 'iptm' else 'ptms'
    score_label = 'ipTM'  if SCORE_TYPE == 'iptm' else 'pTM'
    auc_label   = 'AP' if PRC else 'AUC'

    candidates = glob.glob(os.path.join(WORKING_DIR,
                                        f"MutPredPPI_{TARGET_DATASET}_megascale_all_detailed_results.pkl"))
    if not candidates:
        print(f"ERROR: MutPredPPI_{TARGET_DATASET}_megascale_all_detailed_results.pkl not found")
        return None

    iptm_file = os.path.join(WORKING_DIR, f'iptm_{TARGET_DATASET}_gcv_splits.pkl')
    if not os.path.exists(iptm_file):
        print(f"ERROR: ipTM file not found: {iptm_file}"); return None

    detailed_results      = load_pkl(candidates[0])
    iptm_detailed_results = load_pkl(iptm_file)

    first_iter  = list(iptm_detailed_results['iterations'].keys())[0]
    first_fold  = list(iptm_detailed_results['iterations'][first_iter]['folds'].keys())[0]
    sample_fold = iptm_detailed_results['iterations'][first_iter]['folds'][first_fold]
    has_complex_ids = 'complex_ids' in sample_fold.get('class_1', {})
    has_ptms        = 'ptms'        in sample_fold.get('class_1', {})

    if not has_complex_ids:
        print("WARNING: pkl missing 'complex_ids'. Re-run the builder to regenerate.")
    if not has_ptms and SCORE_TYPE == 'ptm':
        print("WARNING: pkl missing 'ptms'. Re-run the builder to regenerate.")
        return None

    print(f"\n{'='*60}")
    print(f"Method     : {METHOD_DISPLAY_NAME}")
    print(f"Dataset    : {dataset_to_display_name.get(TARGET_DATASET, TARGET_DATASET)}")
    print(f"Score      : {score_label}  (key='{score_key}')")
    print(f"Plot mode  : {PLOT_MODE}")

    if SAVE_PLOTS:
        os.makedirs(os.path.join(WORKING_DIR, SAVE_DIR), exist_ok=True)

    tag       = f"{PLOT_MODE}_{SCORE_TYPE}_{TARGET_DATASET}"
    save_path = os.path.join(WORKING_DIR, SAVE_DIR, f"iptm_{tag}.png") if SAVE_PLOTS else None

    if PLOT_MODE in ('scatter', 'spearman_box'):
        if not has_complex_ids:
            print("Cannot run this mode without complex_ids in pkl."); return None

        complex_auc_iptm = build_complex_auc_table(detailed_results, iptm_detailed_results,
                                                    score_key='iptms', prc=PRC)
        complex_auc_ptm  = build_complex_auc_table(detailed_results, iptm_detailed_results,
                                                    score_key='ptms',  prc=PRC)
        complex_auc = complex_auc_iptm if SCORE_TYPE == 'iptm' else complex_auc_ptm

        print(f"\nComplexes with computable {auc_label}:")
        for c in [1, 2, 3]:
            ck    = f'class_{c}'
            n_cx  = len(complex_auc.get(ck, {}))
            n_var = sum(e['n_points'] for e in complex_auc.get(ck, {}).values())
            print(f"  Class {c}: {n_cx} complexes, {n_var} variants")

        if PLOT_MODE == 'scatter':
            plot_complex_scatter(complex_auc, score_label, TARGET_DATASET,
                                 save_path=save_path, prc=PRC)
            return complex_auc

        else:  # spearman_box
            print("\nComputing per-fold Spearman r values...")
            spearman_table = build_fold_spearman_table(
                detailed_results, iptm_detailed_results,
                complex_auc_iptm, complex_auc_ptm, prc=PRC
            )
            for c in [1, 2, 3]:
                ck = f'class_{c}'
                print(f"  Class {c}: {len(spearman_table[ck]['iptm'])} valid fold-iterations")
            save_path_sp = os.path.join(WORKING_DIR, SAVE_DIR,
                                        f"spearman_box_{TARGET_DATASET}.png") if SAVE_PLOTS else None
            plot_spearman_boxplot(spearman_table, TARGET_DATASET, save_path=save_path_sp)
            return spearman_table

    else:
        binned_iptm = build_binned_results(detailed_results, iptm_detailed_results,
                                           score_key='iptms', threshold=IPTM_THRESH, prc=PRC)
        binned_ptm  = build_binned_results(detailed_results, iptm_detailed_results,
                                           score_key='ptms',  threshold=IPTM_THRESH, prc=PRC)
        binned = binned_iptm if SCORE_TYPE == 'iptm' else binned_ptm

        print("\nNOTE: Wilcoxon test on fold-AUCs is a secondary comparison.")
        print("      Bins are highly unbalanced in n (low bin ~10x more variants).")
        print("      Use scatter Spearman r as the primary reported statistic.")
        for score_label_s, binned_s in [('ipTM', binned_iptm), ('pTM', binned_ptm)]:
            print(f"\n{auc_label} by {score_label_s} bin:")
            for c in [1, 2, 3]:
                ck   = f'class_{c}'
                hi   = binned_s['high'][ck]['aucs']
                lo   = binned_s['low'][ck]['aucs']
                hi_n = int(np.mean(binned_s['high'][ck]['n_points'])) if hi else 0
                lo_n = int(np.mean(binned_s['low'][ck]['n_points']))  if lo else 0
                hi_s = f"{np.mean(hi):.4f}±{np.std(hi, ddof=1):.4f} (n={hi_n})" if hi else "—"
                lo_s = f"{np.mean(lo):.4f}±{np.std(lo, ddof=1):.4f} (n={lo_n})" if lo else "—"
                diff = f"  Δ={np.mean(hi)-np.mean(lo):+.4f}" if (hi and lo) else ""
                print(f"  Class {c}:")
                print(f"    {score_label_s} > {IPTM_THRESH}: {hi_s}")
                print(f"    {score_label_s} ≤ {IPTM_THRESH}: {lo_s}{diff}")

        if PLOT_MODE == 'curve':
            plot_iptm_curves(binned, score_label, TARGET_DATASET,
                             save_path=save_path, prc=PRC)
            return binned
        elif PLOT_MODE == 'boxplot':
            save_path = os.path.join(WORKING_DIR, SAVE_DIR,
                                     f"iptm_boxplot_iptm_ptm_{TARGET_DATASET}.png") if SAVE_PLOTS else None
            plot_iptm_boxplots(binned_iptm, binned_ptm, TARGET_DATASET,
                               save_path=save_path, prc=PRC)
            return {'iptm': binned_iptm, 'ptm': binned_ptm}
        else:
            raise ValueError(f"Unknown PLOT_MODE: {PLOT_MODE!r}")


# %% Run ipTM analysis
if __name__ == '__main__':
    iptm_results = main_iptm_analysis()
