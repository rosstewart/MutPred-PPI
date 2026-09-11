#!/usr/bin/env bash
# Minimal, fast, end-to-end MutPred-PPI inference example.
#
# Runs the real public 3-step inference pipeline (src/inference/) on 3
# hardcoded protein pairs / variants using precomputed AlphaFold3 structures
# already shipped with the repo (converted from datasets/af3_structures/ to
# mmCIF here since the pipeline's contact-graph step requires mmCIF input).
#
# Usage:
#   bash src/inference/example/run_example.sh [--device cpu|cuda:0]
#
# Takes well under a minute on GPU, a few minutes on CPU.

set -euo pipefail

DEVICE="cuda:0"
if [[ "${1:-}" == "--device" ]]; then
    DEVICE="$2"
fi

# resolve repo root (this script lives at src/inference/example/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EXAMPLE_DIR="${SCRIPT_DIR}"

# Preflight.  Without this, a user who forgot to activate the environment gets a
# bare "ModuleNotFoundError: No module named 'h5py'" from inside step 3.
missing=""
for mod in torch torch_geometric h5py numpy pandas transformers; do
    python -c "import ${mod}" 2>/dev/null || missing="${missing} ${mod}"
done
if [[ -n "${missing}" ]]; then
    echo "ERROR: the active Python is $(command -v python)" >&2
    echo "       and is missing:${missing}" >&2
    echo >&2
    echo "  conda activate ppi             # then re-run this script" >&2
    echo "  (environment setup: README.md; data layout: docs/SETUP.md)" >&2
    exit 1
fi

echo "=== MutPred-PPI inference quickstart ==="
echo "Repo root:   ${REPO_ROOT}"
echo "Example dir: ${EXAMPLE_DIR}"
echo "Device:      ${DEVICE}"
echo

cd "${REPO_ROOT}/src/inference"

echo "--- Step 2: generate contact graphs + FASTA from mmCIF structures ---"
python 01_make_contact_graphs_and_fasta.py \
    "${EXAMPLE_DIR}" \
    "${EXAMPLE_DIR}/af3_models" \
    "${EXAMPLE_DIR}/test_variants.tsv" \
    1

echo
echo "--- Step 3: run MutPred-PPI inference (primary model) ---"
python 02_run_mutpred-ppi_inference.py \
    "${EXAMPLE_DIR}" \
    --device "${DEVICE}"

echo
echo "--- Comparing against the committed reference ---"
# Compare, do not overwrite: expected_output/ is the reference this example exists
# to check against, so copying over it would destroy the only drift detector.
REF="${EXAMPLE_DIR}/expected_output/MutPred-PPI_preds.tsv"
GOT="${EXAMPLE_DIR}/results/MutPred-PPI_preds.tsv"
if [[ ! -f "${REF}" ]]; then
    echo "no reference at ${REF}; writing this run as the reference"
    mkdir -p "${EXAMPLE_DIR}/expected_output"; cp "${GOT}" "${REF}"
elif diff <(sort "${REF}") <(sort "${GOT}") >/dev/null; then
    echo "MATCH - predictions are identical to the committed reference"
else
    echo "DIFFERS from the committed reference:" >&2
    diff <(sort "${REF}") <(sort "${GOT}") >&2 || true
    echo >&2
    echo "Small deviations can come from a different torch/CUDA build." >&2
    echo "Large ones mean something is wrong -- do not ignore." >&2
    exit 1
fi

echo
echo "Predictions: ${GOT}"
cat "${GOT}"
