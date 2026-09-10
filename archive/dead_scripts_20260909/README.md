# Archived scripts (2026-09-09)

| Script | Why archived |
|---|---|
| `train_on_subset.py` (was `examples/training_quickstart/`) | Imported `_build_emb_dict` / `_load_graphs` / `_gather_labels_pos_neg` from `mutpred_ppi_cv.py`, all removed when that module was reduced to `train_fold`; rewriting it against the canonical tables would make a deliberately self-contained quickstart depend on the undistributed `mapped090826` rows/graph-store/ProtT5 caches, so it was retired instead. |
