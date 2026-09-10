#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import glob
import os
import re

# --- repo-relative path resolution (see src/paths.py) ---
from paths import VARCHAMP_BLIND_TEST_DIR  # noqa: E402
from utils.blind_test_common import load_class_arrays  # noqa: E402

WORKING_DIR = str(VARCHAMP_BLIND_TEST_DIR)
SAVE_PLOTS = True
SAVE_DIR = os.path.join(WORKING_DIR, "roc_plots")
FIGURE_DPI = 100
SAVE_DPI = 300

METHOD_DISPLAY_NAMES = {
    # VarChAMP blind test method keys (train=sahni_fragoza_mapped090826,
    # test=varchamp_all_mapped090826 -- see run_varchamp_blind_test.py)
    "MutPred-PPI (megascale_all, all-data) (varchamp_blind_test)":        "MutPred-PPI",
    "MutPred-PPI (sahni, megascale_all, all-data) (varchamp_blind_test)": "MutPred-PPI (Sahni only)",
    "eSIG-Net (Sahni+Fragoza train) (varchamp_blind_test)":               "eSIG-Net",
    "SWING (Sahni+Fragoza train) (varchamp_blind_test)":                  "SWING (Blind-Test)",
    "SWING (test pretrain, Sahni+Fragoza train) (varchamp_blind_test)":   "SWING (Test Pretrain)",
    "MutPPI (Sahni+Fragoza train) (varchamp_blind_test)":                 "MutPPI",
    "MutPPIPlus (Sahni+Fragoza train) (varchamp_blind_test)":             "MutPPI+",
    "PPLM_seq_diff (Sahni+Fragoza train) (varchamp_blind_test)":          "PPLM (seq diff)",
    "PPLM_site_diff (Sahni+Fragoza train) (varchamp_blind_test)":         "PPLM (site diff)",
    "MINT_seq_diff (Sahni+Fragoza train) (varchamp_blind_test)":          "MINT (seq diff)",
    "MINT_site_diff (Sahni+Fragoza train) (varchamp_blind_test)":         "MINT (site diff)",
    "SAAMBE-3D (Sahni+Fragoza train) (varchamp_blind_test)":              "SAAMBE-3D",
    "MutPred2 (varchamp_blind_test)":                                     "MutPred2",
}

COLORS = {
    # Matching roc_plots.py GCV figure colors exactly
    "MutPred-PPI":              "#1f77b4",  # blue
    "MutPred-PPI (Sahni only)": "#7b3294",  # dark purple (S2 only, distinct from all main-fig methods)
    "eSIG-Net":                 "#9467bd",  # purple
    "SWING (Blind-Test)":       "#ff7f0e",  # orange
    "SWING (Test Pretrain)":    "#d62728",  # red
    "MutPPI":                   "#17becf",  # teal
    "MutPPI+":                  "#bcbd22",  # yellow-green
    "MutPred2":                 "#2ca02c",  # green
    "PPLM (seq diff)":          "#4b5563",  # dark gray
    "PPLM (site diff)":         "#762a83",  # dark purple
    "MINT (seq diff)":          "#66c2a5",  # light green
    "MINT (site diff)":         "#1b7837",  # dark green
    "SAAMBE-3D":                "#8c564b",  # brown
}

LINE_STYLES = {
    # Fixed (non-GCV) predictors — dashed or dotted to distinguish from trained models
    "MutPred2":  ":",   # dotted
    "SAAMBE-3D": "--",  # dashed
    "MutPPI":    "--",
    "MutPPI+":   "--",
}

# Training-set comparison: all-trained MutPred-PPI models on VCFP blind test (no 10-fold)
TRAINING_SET_COMPARISON_METHODS = [
    "MutPred-PPI (megascale_all, all-data) (varchamp_blind_test)",
    "MutPred-PPI (sahni, megascale_all, all-data) (varchamp_blind_test)",
]

# Set to a list of method keys to restrict which methods appear; None = all in display names
METHODS_TO_COMPARE = [
    "MutPred-PPI (megascale_all, all-data) (varchamp_blind_test)",
    "eSIG-Net (Sahni+Fragoza train) (varchamp_blind_test)",
    "SWING (Sahni+Fragoza train) (varchamp_blind_test)",
    "SWING (test pretrain, Sahni+Fragoza train) (varchamp_blind_test)",
    "MutPPI (Sahni+Fragoza train) (varchamp_blind_test)",
    "MutPPIPlus (Sahni+Fragoza train) (varchamp_blind_test)",
    "PPLM_seq_diff (Sahni+Fragoza train) (varchamp_blind_test)",
    "PPLM_site_diff (Sahni+Fragoza train) (varchamp_blind_test)",
    "MINT_seq_diff (Sahni+Fragoza train) (varchamp_blind_test)",
    "MINT_site_diff (Sahni+Fragoza train) (varchamp_blind_test)",
    "SAAMBE-3D (Sahni+Fragoza train) (varchamp_blind_test)",
    "MutPred2 (varchamp_blind_test)",
]


# Methods with no Sahni+Fragoza training component. C1/C2/C3 is defined by
# training-set membership, so the stratification carries no meaning for them:
# they are scored on the pooled test set and drawn identically in all three panels.
STRATIFICATION_INDEPENDENT_METHODS = {
    "MutPred2 (varchamp_blind_test)",
}


def extract_method_name(filepath):
    fname = os.path.basename(filepath)
    return re.sub(r'_c[123]_(labels|preds|vt_ids)\.npy$', '', fname)


def load_method_data(method_name, directory):
    data = {}
    for c in [1, 2, 3]:
        loaded = load_class_arrays(method_name, c, directory, require_vt_ids=False)
        if loaded is None:
            continue
        preds, labels, _vt_ids = loaded
        if len(preds) == 0:
            print(f"Warning: {method_name} c{c} predictions are empty — skipping.")
            continue
        if (preds.max() - preds.min()) < 1e-5:
            print(f"Warning: {method_name} c{c} predictions are constant ({preds[0]:.4f}) — skipping (degenerate).")
            return {}
        data[f"labels_c{c}"] = labels
        data[f"preds_c{c}"]  = preds

    if data and method_name in STRATIFICATION_INDEPENDENT_METHODS:
        present = [c for c in [1, 2, 3] if f"labels_c{c}" in data]
        labels  = np.concatenate([data[f"labels_c{c}"] for c in present])
        preds   = np.concatenate([data[f"preds_c{c}"]  for c in present])
        for c in [1, 2, 3]:
            data[f"labels_c{c}"] = labels
            data[f"preds_c{c}"]  = preds
        data["pooled"] = True
    return data


def plot_comparison(methods_data, save_name="roc_varchamp_blind_test.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=FIGURE_DPI)
    class_labels = ["C1 (both in train)", "C2 (one in train)", "C3 (neither in train)"]

    for class_idx, c in enumerate([1, 2, 3]):
        ax = axes[class_idx]
        auc_list = []

        for method_name, data in methods_data.items():
            display = METHOD_DISPLAY_NAMES.get(method_name)
            if display is None:
                continue
            labels_key = f"labels_c{c}"
            preds_key  = f"preds_c{c}"
            if labels_key not in data or preds_key not in data:
                continue

            y_true   = data[labels_key]
            y_scores = data[preds_key]
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            color = COLORS.get(display, "#333333")
            lw = 2.5 if "MutPred-PPI" == display else 2
            ls = LINE_STYLES.get(display, "-")
            auc_list.append((roc_auc, display, color, len(y_scores), ls, lw, fpr, tpr))

        auc_list.sort(reverse=True, key=lambda x: x[0])

        for roc_auc, display, color, n, ls, lw, fpr, tpr in auc_list:
            ax.plot(fpr, tpr, color=color, lw=lw, linestyle=ls, alpha=0.85)

        ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5)

        legend_labels  = [f"{name} (n={n}, AUC={a:.3f})" for a, name, _, n, _, _, _, _ in auc_list]
        legend_handles = [plt.Line2D([0], [0], color=col, lw=lw, linestyle=ls)
                          for _, _, col, _, ls, lw, _, _ in auc_list]

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        if class_idx == 0:
            ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(class_labels[class_idx], fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(legend_handles, legend_labels, loc="lower right", fontsize=9)

    plt.tight_layout()

    if SAVE_PLOTS:
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR, save_name)
        plt.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_training_set_comparison(methods_data, save_name="roc_varchamp_blind_test_training_comparison.png"):
    """S2: compare sahni-only vs sahni+fragoza model on VCFP blind test."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=FIGURE_DPI)
    class_labels = ["C1 (both in train)", "C2 (one in train)", "C3 (neither in train)"]

    for class_idx, c in enumerate([1, 2, 3]):
        ax = axes[class_idx]
        auc_list = []

        for method_name, data in methods_data.items():
            display = METHOD_DISPLAY_NAMES.get(method_name)
            if display is None:
                continue
            labels_key = f"labels_c{c}"
            preds_key  = f"preds_c{c}"
            if labels_key not in data or preds_key not in data:
                continue

            y_true   = data[labels_key]
            y_scores = data[preds_key]
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            color = COLORS.get(display, "#333333")
            lw = 2.5
            ls = "-" if "Sahni only" not in display else "--"
            auc_list.append((roc_auc, display, color, len(y_scores), ls, lw, fpr, tpr))

        auc_list.sort(reverse=True, key=lambda x: x[0])

        for roc_auc, display, color, n, ls, lw, fpr, tpr in auc_list:
            ax.plot(fpr, tpr, color=color, lw=lw, linestyle=ls, alpha=0.85)

        ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5)

        legend_labels  = [f"{name} (n={n}, AUC={a:.3f})" for a, name, _, n, _, _, _, _ in auc_list]
        legend_handles = [plt.Line2D([0], [0], color=col, lw=lw, linestyle=ls)
                          for _, _, col, _, ls, lw, _, _ in auc_list]

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        if class_idx == 0:
            ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(class_labels[class_idx], fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(legend_handles, legend_labels, loc="lower right", fontsize=9)

    plt.suptitle("VarChAMP (VCFP): Training Set Comparison", fontsize=15)
    plt.tight_layout()

    if SAVE_PLOTS:
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR, save_name)
        plt.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def load_all_methods(method_list, directory):
    label_files = glob.glob(os.path.join(directory, "*_c?_labels.npy"))
    all_methods = {extract_method_name(f) for f in label_files}

    methods_to_use = (
        [m for m in method_list if m in all_methods]
        if method_list else sorted(all_methods)
    )
    missing = [m for m in (method_list or []) if m not in all_methods]
    if missing:
        print(f"Warning: not found: {missing}")

    all_data = {}
    for method in methods_to_use:
        data = load_method_data(method, directory)
        if data:
            all_data[method] = data
    return all_data


def run_analysis():
    if not os.path.exists(WORKING_DIR):
        raise FileNotFoundError(f"Results directory not found: {WORKING_DIR}")

    all_data = load_all_methods(METHODS_TO_COMPARE, WORKING_DIR)

    print(f"\nMethods loaded ({len(all_data)}):")
    for m in all_data:
        display = METHOD_DISPLAY_NAMES.get(m, m)
        sizes = {c: len(all_data[m].get(f"labels_c{c}", [])) for c in [1,2,3]}
        aucs  = {}
        for c in [1,2,3]:
            if f"labels_c{c}" in all_data[m]:
                fpr, tpr, _ = roc_curve(all_data[m][f"labels_c{c}"], all_data[m][f"preds_c{c}"])
                aucs[c] = auc(fpr, tpr)
        auc_str = "  ".join(f"C{c}={aucs[c]:.3f}" for c in [1,2,3] if c in aucs)
        print(f"  {display}: n={sizes}  {auc_str}")

    if all_data:
        plot_comparison(all_data)

    # S2: training set comparison (runs if sahni-only model results are present)
    comparison_data = load_all_methods(TRAINING_SET_COMPARISON_METHODS, WORKING_DIR)
    if len(comparison_data) == len(TRAINING_SET_COMPARISON_METHODS):
        print("\nGenerating S2 training-set comparison figure...")
        plot_training_set_comparison(comparison_data)
    else:
        missing_ts = [m for m in TRAINING_SET_COMPARISON_METHODS if m not in comparison_data]
        print(f"\nSkipping S2 training-set comparison: missing {missing_ts}")

    return all_data


if __name__ == "__main__":
    run_analysis()
