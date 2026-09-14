#!/usr/bin/env bash
#
# Create external/ -- symlinks to the large, machine-local trees the repo cannot
# ship: unpublished VarChAMP data, multi-GB embedding and contact-graph sets, and
# third-party model checkpoints.
#
# Code resolves these through paths.EXTERNAL_DIR, so a machine that has the data
# somewhere else only needs to re-run this script with MUTPRED_DATA_ROOT set --
# no source edits.  Links that cannot be satisfied are reported and skipped, so
# a partial environment still works for whatever it does have.
#
# Usage:
#   bash scripts/link_external.sh                 # default roots
#   MUTPRED_DATA_ROOT=/other/path bash scripts/link_external.sh
#
# Two links sit outside MUTPRED_DATA_ROOT on the machine this was written for
# and have their own overrides, so no path in this script is machine-specific:
#   MUTPRED_AFDB            AlphaFold DB human monomer PDBs
#   MUTPRED_SWING_DOC2VEC   SWING's per-fold Doc2Vec model
#
# What does NOT belong here: anything small enough to live in datasets/.  Those
# are copied, not linked, so the analysis layer is self-contained.

set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
DATA_ROOT="${MUTPRED_DATA_ROOT:-$(dirname "$REPO")}"
EXT="$REPO/external"
mkdir -p "$EXT"

echo "repo:      $REPO"
echo "data root: $DATA_ROOT"
echo

link() {  # link <target> <name>
  local target="$1" name="$2"
  if [ ! -e "$target" ]; then
    printf "  %-28s MISSING  %s\n" "$name" "$target"
    return
  fi
  ln -sfn "$target" "$EXT/$name"
  printf "  %-28s %-8s %s\n" "$name" "$(du -sh "$target" 2>/dev/null | cut -f1)" "$target"
}

echo "== training / evaluation data =="
link "$DATA_ROOT/swing_train"                    swing_train
link "$DATA_ROOT/home/sahni/af3_graphs"          sahni_af3_graphs
link "$DATA_ROOT/home/data_interaction_loss"     data_interaction_loss
link "$DATA_ROOT/sahni_wt_and_vt_t5.pkl"         sahni_wt_and_vt_t5.pkl
link "$DATA_ROOT/megascale_preprocessed"         megascale_preprocessed
echo

echo "== unpublished VarChAMP (IGVF; local only, never distributed) =="
link "$DATA_ROOT/varchamp1p"                     varchamp1p
link "$DATA_ROOT/cava"                           cava
link "$DATA_ROOT/varchamp_pooled"                varchamp_pooled
link "$DATA_ROOT/2026/graphs"                    vc2026_graphs
link "$DATA_ROOT/2026/all_labeled_prott5_embeddings.pkl" vc2026_t5_embeddings.pkl
echo

echo "== structures for the structural comparators =="
link "$DATA_ROOT/three_datasets_af3_models/pdbs" af3_pdbs
link "$DATA_ROOT/sahni_pdbs"                     sahni_pdbs
# AlphaFold DB human monomers. Override with MUTPRED_AFDB if they live
# elsewhere; this is a public download, not something we distribute.
link "${MUTPRED_AFDB:-$DATA_ROOT/alphafold_v4_human}" alphafold_v4_human
echo

# SWING's Doc2Vec is data-dependent (retrained per fold for the blind-test
# variant), so it is a model artifact rather than a shippable input -- linked,
# never distributed.  Only the leaky "Test Pretrain" variant loads it.
link "${MUTPRED_SWING_DOC2VEC:-$DATA_ROOT/SWING_sahni_fragoza_doc2vec.model}" SWING_sahni_fragoza_doc2vec.model
echo

echo "== third-party model checkpoints (see docs/REPRODUCING_ANALYSES.md) =="
link "$DATA_ROOT/2026/mint"                      mint
link "$DATA_ROOT/2026/PPLM"                      pplm
link "$DATA_ROOT/2026/eSIG-Net"                  esignet
echo

echo "== regenerable caches (see docs/REPRODUCING_ANALYSES.md for the commands) =="
link "$DATA_ROOT/mutpred_ppi_data"               caches
link "$DATA_ROOT/biogrid"                        biogrid
echo

broken=$(find "$EXT" -maxdepth 1 -xtype l 2>/dev/null | wc -l)
total=$(find "$EXT" -maxdepth 1 -type l 2>/dev/null | wc -l)
echo "external/: $total links, $broken broken"
[ "$broken" -gt 0 ] && echo "  (broken links are fine if you do not run the analyses that need them)"
exit 0
