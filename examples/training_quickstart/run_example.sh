#!/usr/bin/env bash
# Minimal, fast, real end-to-end MutPred-PPI TRAINING example.
#
# Trains a MutPred-PPI checkpoint from scratch (ablation="scratch") for 1
# epoch on a tiny (40-row) real subset of the Sahni+Fragoza training data,
# using the real GAT_mut_processor model architecture, and verifies a .pt
# checkpoint is produced.
#
# Usage:
#   bash examples/training_quickstart/run_example.sh [--device cpu|cuda:0] [--epochs 1]
#
# Takes well under a minute on GPU (ProtT5 embedding generation dominates),
# a few minutes on CPU.

set -euo pipefail

DEVICE="cuda:0"
EPOCHS=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXAMPLE_DIR="${SCRIPT_DIR}"

# Preflight -- same rationale as the inference quickstart: without this a user
# who forgot to activate the environment gets an opaque ModuleNotFoundError
# several steps in, after the slow embedding pass has already run.
missing=""
for mod in torch torch_geometric h5py numpy pandas transformers sklearn; do
    python -c "import ${mod}" 2>/dev/null || missing="${missing} ${mod}"
done
if [[ -n "${missing}" ]]; then
    echo "ERROR: the active Python is $(command -v python)" >&2
    echo "       and is missing:${missing}" >&2
    echo >&2
    echo "  conda activate mutpred-ppi     # then re-run this script" >&2
    echo "  (see docs/SETUP.md and requirements.txt)" >&2
    exit 1
fi

echo "=== MutPred-PPI training quickstart ==="
echo "Repo root:   ${REPO_ROOT}"
echo "Example dir: ${EXAMPLE_DIR}"
echo "Device:      ${DEVICE}"
echo "Epochs:      ${EPOCHS}"
echo

cd "${REPO_ROOT}/src/inference"

echo "--- Step 1: generate contact graphs + FASTA from mmCIF structures (real public pipeline script) ---"
python 01_make_contact_graphs_and_fasta.py \
    "${EXAMPLE_DIR}" \
    "${EXAMPLE_DIR}/af3_models" \
    "${EXAMPLE_DIR}/train_variants.tsv" \
    4

echo
echo "--- Step 2: fix up pos/neg labels using true Y2H_score from sahni_fragoza_train_subset.csv ---"
python "${EXAMPLE_DIR}/fix_labels.py"

echo
echo "--- Step 3: generate ProtT5 embeddings for the training subset (real ProtT5 loader utils) ---"
python "${EXAMPLE_DIR}/generate_embeddings.py" --device "${DEVICE}"

echo
echo "--- Step 4: train MutPred-PPI (GAT_mut_processor, ablation=scratch) for ${EPOCHS} epoch(s) ---"
python "${EXAMPLE_DIR}/train_on_subset.py" --device "${DEVICE}" --epochs "${EPOCHS}"

echo
echo "Done. Checkpoint written to:"
echo "  ${EXAMPLE_DIR}/output/MutPred-PPI_quickstart_scratch.pt"
ls -la "${EXAMPLE_DIR}/output/"
