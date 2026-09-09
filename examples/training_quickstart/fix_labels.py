#!/usr/bin/env python
"""Fix up the .interaction_loss_pos files written by
src/inference/01_make_contact_graphs_and_fasta.py so they encode the real
Y2H_score (disrupted/maintained) label, split into proper _pos / _neg files.

Why this is needed: 01_make_contact_graphs_and_fasta.py is the public
inference pipeline script; it has no notion of a ground-truth label, so it
writes every variant's mutation position into a single ".interaction_loss_pos"
file regardless of true label (that file just marks "this position is
mutated", used at inference time to locate the mutation site). The training
data loaders in src/evaluation/mutpred_ppi_cv.py (_gather_labels_pos_neg)
expect two separate files: ".interaction_loss_pos" for variants that disrupt
the interaction (Y2H_score == 1) and ".interaction_loss_neg" for variants that
maintain it (Y2H_score == 0). This script does that split, using the true
labels from sahni_fragoza_train_subset.csv (a real subset of
datasets/train_eval/sahni_fragoza_train.csv). It does not modify
01_make_contact_graphs_and_fasta.py itself.
"""
import csv
import os
from pathlib import Path

csv.field_size_limit(10**9)

_HERE = Path(__file__).resolve().parent
GRAPH_DIR = str(_HERE / "af3_graphs")
CSV_PATH = str(_HERE / "sahni_fragoza_train_subset.csv")


def main():
    true_label = {}
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            key = (row["refseq_id"], row["partner"])
            true_label.setdefault(key, {})[row["Mutation"]] = row["Y2H_score"]

    n_pos = n_neg = 0
    for (id_a, id_b), mut_map in true_label.items():
        pos_path = f"{GRAPH_DIR}/{id_a}_{id_b}.interaction_loss_pos"
        neg_path = f"{GRAPH_DIR}/{id_a}_{id_b}.interaction_loss_neg"
        if not os.path.exists(pos_path):
            print(f"WARNING: missing {pos_path}, skipping")
            continue

        with open(pos_path) as f:
            lines = [line.rstrip("\n") for line in f if line.strip()]

        pos_lines, neg_lines = [], []
        for line in lines:
            _idx_str, variant_0based = line.split("\t")
            wt, mt = variant_0based[0], variant_0based[-1]
            pos0 = int(variant_0based[1:-1])
            mutation_1based = f"{wt}{pos0 + 1}{mt}"
            label = mut_map.get(mutation_1based)
            assert label is not None, (id_a, id_b, mutation_1based, mut_map)
            (pos_lines if label == "1" else neg_lines).append(line)

        with open(pos_path, "w") as f:
            f.write("\n".join(pos_lines) + ("\n" if pos_lines else ""))
        with open(neg_path, "w") as f:
            f.write("\n".join(neg_lines) + ("\n" if neg_lines else ""))

        n_pos += len(pos_lines)
        n_neg += len(neg_lines)

    print(f"Relabeled: {n_pos} disrupted (pos), {n_neg} maintained (neg)")


if __name__ == "__main__":
    main()
