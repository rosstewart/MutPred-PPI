"""Machinery shared by more than one area of the codebase.

Nothing in here belongs to `evaluation/`, `training/`, `analysis/`,
`inference/` or `variant_db_inference/` alone -- each module is imported by at
least two of them, which is the only reason it is here rather than next to its
single consumer.

    gcv_common        canonical dataset configs, row/split loading, clustering
                      and the shared GCV runner
    structures        contact-graph store and AF3 structure lookup, keyed on
                      sequence content
    mutpred_ppi_data  graph + ProtT5 tensors for MutPred-PPI, built from the
                      canonical tables
    mutpred_ppi_cv    the MutPred-PPI training loop (`train_fold`), shared by
                      the GCV runner and the shipped-model trainer
"""
