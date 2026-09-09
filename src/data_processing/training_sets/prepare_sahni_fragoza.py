#!/usr/bin/env python
"""Prepare the Sahni+Fragoza combined training dataset (SWING-style).

Reads .pos/.neg interaction files and a FASTA of WT/VT sequences, validates
variant positions, optionally trains a Doc2Vec embedding + XGBoost classifier
with group cross-validation, and saves the final training CSV and CV results.

Usage:
    python prepare_sahni_fragoza.py \
        --pos-file swing_train/swing_train.pos \
        --neg-file swing_train/swing_train.neg \
        --fasta swing_train/swing_train_wt_and_vt.fasta \
        --cv-splits cv_splits/sahni_fragoza_train_fold_splits.pkl \
        --vt-ids cv_splits/sahni_fragoza_train_all_vt_ids.pkl \
        --pair-test-classes cv_splits/swing_train_pair_test_classes.npy \
        --output-dir $MUTPRED_DATA_ROOT \
        --train-doc2vec \
        --train-xgboost
"""

import argparse
import os
import pickle
import random
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from Bio import SeqIO

random.seed(42)


# ---------------------------------------------------------------------------
# SWING window-encoding helpers
# ---------------------------------------------------------------------------

AA_SCORES = {
    "A": 8.1, "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5,
    "E": 12.3, "Q": 10.5, "G": 9.0,  "H": 10.4, "I": 5.2,
    "L": 4.9, "K": 11.3, "M": 5.7,  "F": 5.2,  "P": 8.0,
    "S": 9.2, "T": 8.6,  "W": 5.4,  "Y": 6.2,  "V": 5.9,
}

def build_aa_score_dict():
    AAs = list(AA_SCORES.keys())
    d = {}
    for i in range(len(AAs)):
        for j in range(len(AAs) - i):
            pair = AAs[i] + AAs[j + i]
            score = round(abs(AA_SCORES[AAs[i]] - AA_SCORES[AAs[j + i]]))
            d[pair] = score
            d[pair[::-1]] = score
    return d


def get_window_encodings(df, window_k=1, pos_colname="Position",
                         mutseq_colname="Mutated_Seq (unless WT)",
                         intseq_colname="Interactor_Seq",
                         aa_score_dict=None, padding_score=9):
    if aa_score_dict is None:
        aa_score_dict = build_aa_score_dict()
    total_encodings = []
    for i in tqdm(df.index):
        pos = df[pos_colname].iloc[i] - 1
        mut_window = df[mutseq_colname].iloc[i][pos - window_k: pos + window_k + 1]
        interactor = df[intseq_colname].iloc[i]
        PPI_encoding = ""
        for its in range(len(interactor)):
            window_scores = ""
            for k in range(len(mut_window)):
                try:
                    pair = mut_window[k] + interactor[k + its]
                    score = aa_score_dict[pair]
                except Exception:
                    score = padding_score
                window_scores += str(score)
            PPI_encoding += window_scores
        total_encodings.append(PPI_encoding)
    return total_encodings


def get_kmers_str(encoding_scores, k=7, padding_score=9):
    padding = {str(padding_score) * (i + 1) for i in range(k)}
    padding.add(str(padding_score))
    kmers = []
    for ppi_score in tqdm(encoding_scores):
        int_kmers = [ppi_score[j: j + k]
                     for j in range(len(ppi_score) - k + 1)
                     if ppi_score[j: j + k] not in padding]
        kmers.append(int_kmers)
    return kmers


def get_corpus(matrix):
    import gensim
    for i, doc in enumerate(matrix):
        yield gensim.models.doc2vec.TaggedDocument(doc, [i])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_interaction_files(pos_file, neg_file):
    pos_df = pd.read_csv(pos_file, header=None, sep="\t")
    pos_df.columns = ["refseq_id", "Mutation", "partner"]
    neg_df = pd.read_csv(neg_file, header=None, sep="\t")
    neg_df.columns = ["refseq_id", "Mutation", "partner"]
    df = pd.concat([pos_df, neg_df], ignore_index=True)
    labels = [1] * len(pos_df) + [0] * len(neg_df)
    df["Y2H_score"] = labels
    return df


def load_fasta_dict(fasta_path):
    """Load FASTA, converting 0-based variant positions to 1-based in the key."""
    fasta_dict = {
        record.description.strip(): str(record.seq)
        for record in SeqIO.parse(fasta_path, "fasta")
    }
    # Convert 0-based variant labels to 1-based
    updated = {}
    for key in list(fasta_dict.keys()):
        if " " in key:
            refseq_id, zero_based_variant = key.split(" ")
            one_based_variant = (
                zero_based_variant[0]
                + str(int(zero_based_variant[1:-1]) + 1)
                + zero_based_variant[-1]
            )
            updated[f"{refseq_id} {one_based_variant}"] = fasta_dict[key]
        else:
            updated[key] = fasta_dict[key]
    return updated


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------

def build_dataset(df, fasta_dict):
    """Attach WT/VT sequences and parse variant positions."""
    indices_to_drop = []
    new_info = []
    for i, row in df.iterrows():
        refseq_id, variant, partner = row["refseq_id"], row["Mutation"], row["partner"]
        vt_id = f"{refseq_id} {variant}"
        if refseq_id not in fasta_dict or vt_id not in fasta_dict or partner not in fasta_dict:
            indices_to_drop.append(i)
            new_info.append((None,) * 7)
            continue
        target_seq = fasta_dict[refseq_id]
        interactor_seq = fasta_dict[partner]
        mutated_seq = fasta_dict[vt_id]
        before_aa = variant[0]
        position = int(variant[1:-1])
        after_aa = variant[-1]
        if before_aa == after_aa:
            indices_to_drop.append(i)
            new_info.append((None,) * 7)
            continue
        assert target_seq[position - 1] == before_aa and mutated_seq[position - 1] == after_aa
        new_info.append((before_aa, position, after_aa, target_seq, interactor_seq, mutated_seq, "Mutant"))

    new_cols = pd.DataFrame(
        new_info,
        columns=["Before_AA", "Position", "After_AA", "Target_Seq", "Interactor_Seq",
                 "Mutated_Seq (unless WT)", "Type"],
    )
    df = pd.concat([df, new_cols], axis=1)
    df = df.drop(indices_to_drop).reset_index(drop=True)
    df["Position"] = df["Position"].astype(int)
    assert not df.duplicated().any()
    return df


def add_wildtype_rows(df):
    """Append synthetic WT rows (mutated seq = WT) for Doc2Vec pre-training."""
    wt_seqs = []
    for i in df.index:
        mut_seq = df.loc[i]["Mutated_Seq (unless WT)"]
        after_aa = df.loc[i]["After_AA"]
        position = df.loc[i]["Position"] - 1
        if mut_seq[position] == after_aa:
            wt_seqs.append(mut_seq[: position] + after_aa + mut_seq[position + 1:])
        else:
            raise ValueError(f"Position index mismatch at index {i}")
    train_wts = df.copy()
    train_wts["Mutated_Seq (unless WT)"] = wt_seqs
    train_wts["Type"] = "WildType"
    return pd.concat([df, train_wts]).sample(frac=1, random_state=1).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Training (Doc2Vec + XGBoost)
# ---------------------------------------------------------------------------

def train_doc2vec(df_with_wts, dim=128, dm=1, alpha=0.08711, window=6, epochs=52):
    from gensim.models.doc2vec import Doc2Vec

    aa_score_dict = build_aa_score_dict()
    window_encodings = get_window_encodings(df_with_wts, window_k=1, aa_score_dict=aa_score_dict)
    kmers = get_kmers_str(window_encodings, k=7)
    train_corpus = list(get_corpus(kmers))

    model = Doc2Vec(vector_size=dim, min_count=1, alpha=alpha, dm=dm, window=window)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=epochs)
    print("Doc2Vec training done")
    return model, model.dv.vectors


def run_cross_validation(ordered_df, fold_splits, features, labels,
                         n_estimators=375, max_depth=6, learning_rate=0.08966):
    from xgboost import XGBClassifier
    from sklearn import metrics
    from sklearn.metrics import precision_recall_curve, auc

    all_preds, all_labels = [], []
    for fold, train_idx, test_idx in fold_splits:
        X_train = features[train_idx]
        X_test = features[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        xgb = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                             learning_rate=learning_rate)
        xgb.fit(X_train, y_train)
        pred_proba = xgb.predict_proba(X_test)[:, 1]
        all_preds.extend(pred_proba)
        all_labels.extend(y_test)
        auc_score = metrics.roc_auc_score(y_test, pred_proba)
        print(f"Fold {fold} done — AUC-ROC: {auc_score:.4f}")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    roc_auc = metrics.roc_auc_score(all_labels, all_preds)
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    prc_auc = auc(recall, precision)
    print(f"\nOverall AUC-ROC: {roc_auc:.4f}")
    print(f"Overall AUC-PRC: {prc_auc:.4f}")
    return all_preds, all_labels


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load interaction data
    df = load_interaction_files(args.pos_file, args.neg_file)
    print(f"Loaded {len(df)} interactions ({Counter(df.Y2H_score.values)})")

    # 2. Load FASTA
    fasta_dict = load_fasta_dict(args.fasta)

    # 3. Build dataset with sequences
    df = build_dataset(df, fasta_dict)
    print(f"Dataset after sequence validation: {len(df)} rows")

    # 4. Save training CSV
    csv_out = os.path.join(args.output_dir, "sahni_fragoza_train.csv")
    df.to_csv(csv_out, index=False)
    print(f"Saved training data to {csv_out}")

    if not args.train_doc2vec:
        print("Skipping Doc2Vec training (--train-doc2vec not set)")
        return

    # 5. Add WT rows and train Doc2Vec
    df_with_wts = add_wildtype_rows(df)
    d2v_model, all_vecs = train_doc2vec(df_with_wts)

    model_out = os.path.join(args.output_dir, "SWING_sahni_fragoza_doc2vec.model")
    d2v_model.save(model_out)
    print(f"Saved Doc2Vec model to {model_out}")

    df_with_wts["Vectors"] = all_vecs.tolist()
    updated_df = df_with_wts[df_with_wts["Type"] == "Mutant"]

    if not args.train_xgboost:
        print("Skipping XGBoost CV (--train-xgboost not set)")
        return

    if args.cv_splits is None or args.vt_ids is None:
        print("--cv-splits and --vt-ids are required for XGBoost CV; skipping")
        return

    # 6. Load CV splits and align data
    with open(args.cv_splits, "rb") as f:
        fold_splits = pickle.load(f)
    with open(args.vt_ids, "rb") as f:
        vt_ids_splits = pickle.load(f)

    # Convert 0-based vt_ids to 1-based
    vt_ids_splits = [
        f"{v.split(' ')[0]} {v.split(' ')[1][0]}{int(v.split(' ')[1][1:-1]) + 1}{v.split(' ')[1][-1]}"
        for v in vt_ids_splits
    ]

    index_mapping = []
    valid_indices = []
    for i, row in updated_df.iterrows():
        vt_id = f"{row['refseq_id']}-{row['partner']} {row['Mutation']}"
        if vt_id in vt_ids_splits:
            index_mapping.append(vt_ids_splits.index(vt_id))
            valid_indices.append(i)

    assert len(valid_indices) == len(vt_ids_splits), (
        f"Expected {len(vt_ids_splits)} matching IDs, found {len(valid_indices)}"
    )
    updated_df = updated_df.loc[valid_indices].reset_index(drop=True)
    ordered_df = pd.DataFrame(index=range(len(updated_df)), columns=updated_df.columns)
    for old_idx, new_idx in enumerate(index_mapping):
        ordered_df.iloc[new_idx] = updated_df.iloc[old_idx]

    features = np.array(list(ordered_df["Vectors"].values))
    labels = np.array(list(ordered_df["Y2H_score"].values))

    # 7. Run cross-validation
    all_preds, all_labels = run_cross_validation(ordered_df, fold_splits, features, labels)

    preds_out = os.path.join(args.output_dir, "SWING_sahni_fragoza_test_pretrain_preds.npy")
    labels_out = os.path.join(args.output_dir, "SWING_sahni_fragoza_test_pretrain_labels.npy")
    np.save(preds_out, all_preds)
    np.save(labels_out, all_labels)
    print(f"Saved CV predictions to {preds_out}")

    # 8. Optionally evaluate per pair-test class
    if args.pair_test_classes:
        pair_test_classes = np.load(args.pair_test_classes)
        from sklearn import metrics
        from sklearn.metrics import precision_recall_curve, auc as sklearn_auc
        for c in (1, 2, 3):
            mask = pair_test_classes == c
            roc = metrics.roc_auc_score(all_labels[mask], all_preds[mask])
            prec, rec, _ = precision_recall_curve(all_labels[mask], all_preds[mask])
            prc = sklearn_auc(rec, prec)
            print(f"Class {c} (n={mask.sum()}): AUC-ROC={roc:.4f}, AUC-PRC={prc:.4f}")

        # Save per-class results
        results_dir = os.path.join(args.output_dir, "publication/results/sahni_fragoza_cv")
        os.makedirs(results_dir, exist_ok=True)
        for c in (1, 2, 3):
            mask = pair_test_classes == c
            np.save(f"{results_dir}/SWING_sahni_fragoza_test_pretrain_preds_c{c}.npy",
                    all_preds[mask])
            np.save(f"{results_dir}/SWING_sahni_fragoza_test_pretrain_labels_c{c}.npy",
                    all_labels[mask])

    # 9. Train final model on all data
    indices = np.arange(features.shape[0])
    np.random.shuffle(indices)
    from xgboost import XGBClassifier
    xgb_final = XGBClassifier(n_estimators=375, max_depth=6, learning_rate=0.08966)
    xgb_final.fit(features[indices], labels[indices])
    model_json = os.path.join(args.output_dir, "SWING_xgb_sahni_fragoza_all.json")
    xgb_final.save_model(model_json)
    print(f"Saved final XGBoost model to {model_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Prepare Sahni+Fragoza training dataset and train SWING model"
    )
    p.add_argument("--pos-file", required=True,
                   help="Tab-separated file of positive (interacting) examples")
    p.add_argument("--neg-file", required=True,
                   help="Tab-separated file of negative (non-interacting) examples")
    p.add_argument("--fasta", required=True,
                   help="FASTA of WT and VT sequences (0-based variant positions in header)")
    p.add_argument("--cv-splits", default=None,
                   help="Pickle of (fold, train_idx, test_idx) tuples for group CV")
    p.add_argument("--vt-ids", default=None,
                   help="Pickle of ordered variant IDs matching cv-splits")
    p.add_argument("--pair-test-classes", default=None,
                   help=".npy array of pair-test class labels (1/2/3) for stratified reporting")
    p.add_argument("--output-dir", required=True,
                   help="Directory for output files")
    p.add_argument("--train-doc2vec", action="store_true",
                   help="Train Doc2Vec embeddings (slow; ~6 min)")
    p.add_argument("--train-xgboost", action="store_true",
                   help="Run XGBoost group cross-validation (requires --cv-splits and --vt-ids)")
    main(p.parse_args())
