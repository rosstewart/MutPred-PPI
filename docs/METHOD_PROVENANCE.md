# Comparison Method Provenance and Deviations

Every third-party method benchmarked in the paper, the exact upstream revision it came from, and
every way our usage departs from that upstream. Written so a reviewer can audit the comparison
without access to this machine.

Nothing here is a judgement about method quality — it is a record of what was run.

## Pinned upstream revisions

| Method | Repository | Commit | Role |
|---|---|---|---|
| eSIG-Net | `Stephen-Yi-Laboratory/eSIG-Net` | `cd36a4a` | Comparator |
| MINT | `VarunUllanat/mint` | `1294612` | Comparator (embeddings) |
| PPLM | `junliu621/PPLM` | `c2a4d5d` | Comparator (embeddings) |
| SWING | `jishnu-lab/SWING` | `b6c2049` | Comparator |
| SAAMBE-3D | `delphi001/SAAMBE-3D` | `182a274` | Comparator |
| DDMut-PPI | REST API, `biosig.lab.uq.edu.au/ddmut_ppi` | n/a | Comparator (excluded, see below) |
| MutPred2 | `vpejaver/mutpred2` | `c92bf33` | Comparator (scores imported) |
| SPURS | `luo-group/SPURS` | `bd33651` | **Not benchmarked** — MegaScale data source only |
| RaSP | `KULL-Centre/_2022_ML-ddG-Blaabjerg` | `f587f0d` | **Not benchmarked** — provenance of one ablation checkpoint |

Environment the published numbers were produced under: `scikit-learn 1.6.1`, `numpy 1.26.4`,
`scipy 1.15.3`, `torch 2.5.1+cu121`, `xgboost 3.0.2`, `gensim 4.4.0`. scikit-learn matters: it
drives the MINT and PPLM heads.

---

## eSIG-Net

**What we take from upstream.** `backbones/sdnn/sdnn_model.py` is loaded directly from the upstream
checkout by file path (`src/evaluation/predictors/esignet.py`), and is byte-identical to its git
HEAD. The architecture is unmodified.

**Training protocol is upstream's, unchanged.** 8 epochs, `lr=1e-3`, `batch_size=32`,
`dropout_p=0.3`, `pair_weight=0.05`, `discrim_weight=1.0`, Adam `weight_decay=1e-4`, `seed=42` —
all read from upstream's own `config_train.yaml`. We did not tune them.

**Disclosed asymmetry.** Running eSIG-Net as published means it gets a fixed 8 epochs with no
validation split, no early stopping, no best-epoch restore, and unweighted cross-entropy.
MutPred-PPI is trained with up to 100 epochs, an inner `GroupKFold` validation split,
`ReduceLROnPlateau`, early stopping, best-state restore, and `pos_weight` class balancing. We
consider using each method's published configuration the fairer reading, but the asymmetry is real
and readers should weigh it.

**Featurization is ours, because upstream ships none.** eSIG-Net publishes the *output* of its
573-dim feature pipeline (upstream's own `datasets/embeddings/sdnn_corrected_ppi.h5`, 1094
proteins × 573 — a path in the eSIG-Net checkout, not in this repo) but not
the code that produced it. `_compute_573` reconstructs it as AAC (20) + Conjoint Triad (343) +
auto-covariance (210). `repro_test/validate_esignet_features.py` audits that reconstruction against
the shipped h5.

Two findings from that audit, both recorded here because they cut in opposite directions:

1. **Conjoint Triad normalization was wrong and has been corrected.** We previously applied
   LightGBM-PPI's `(count − min) / max`; upstream frequency-normalizes.

   | CT block (343 dims) | rowsum | max | mean | frac_zero | p95 |
   |---|---|---|---|---|---|
   | upstream h5 | **1.0000** | 0.10256 | 0.002915 | 0.495 | 0.01115 |
   | ours, before | **44.5134** | 1.00000 | 0.129777 | 0.493 | 0.50000 |
   | ours, after (`count/total`) | **1.0000** | 0.08333 | 0.002915 | 0.493 | 0.01097 |

   CT is 343 of the 573 features, so the old normalization fed the SDNN a block ~45× larger in
   aggregate than upstream's while AAC and AC matched — distorting the block balance the model sees.

   Corrected, and the full 30-seed GCV re-run on `sahni_fragoza`. The old normalization was mildly
   handicapping eSIG-Net; correcting it improves the baseline consistently but slightly:

   | | published (old CT) | corrected (`count/total`) | Δ |
   |---|---|---|---|
   | C1 | 0.8434 ± 0.0077 | 0.8451 ± 0.0142 | **+0.0017** |
   | C2 | 0.7849 ± 0.0111 | 0.7879 ± 0.0102 | **+0.0030** |
   | C3 | 0.6446 ± 0.0219 | 0.6517 ± 0.0358 | **+0.0072** |
   | macro | 0.7576 ± 0.0110 | 0.7616 ± 0.0124 | **+0.0039** |

   Every class improves, and every delta is well inside one standard deviation across seeds. No
   conclusion in the paper changes: MutPred-PPI's C3 (0.7536) remains far above eSIG-Net's corrected
   0.6517. Both sets of numbers are reported so the correction is auditable.

2. **The auto-covariance table is correct as-is and was NOT changed.** Tyrosine carries
   `NCI=117.3, V=0.023599` while every other residue has NCI in [0.003, 0.24] and V in [29, 145].
   This looks like a transposition, but reproducing upstream's shipped features *requires* those
   values. Per-property between-protein variance profile vs the upstream h5:

   | property | upstream | ours (as shipped) | ours (Tyr "corrected") |
   |---|---|---|---|
   | H1 | 0.06793 | 0.06263 | 0.06263 |
   | H2 | 0.06166 | 0.05537 | 0.05537 |
   | **NCI** | **0.03959** | **0.03714** | **0.06891** |
   | P1 | 0.06580 | 0.05992 | 0.05992 |
   | P2 | 0.05229 | 0.04827 | 0.04827 |
   | SASA | 0.06569 | 0.05747 | 0.05747 |
   | V | 0.05731 | 0.05030 | 0.05484 |

   Profile correlation with upstream: **+0.9899 as shipped, −0.2467 "corrected"**. The apparent bug
   is faithful reproduction of upstream's own table. Do not change it.

**Note on the two checkouts.** Two eSIG-Net clones exist on the build machine:
`2026/eSIG-Net` (`cd36a4a`, `Stephen-Yi-Laboratory`) and `home/eSIG-Net` (`dea5d9a`,
`Yilab-texas`). They are the **same upstream code** — every `.py`, `.yaml` and `.sh` is
byte-identical at HEAD; the repository simply moved organisations. The paper loads the pristine
`2026/` clone.

---

## MINT

**Embeddings from upstream, unmodified.** `mint/model/esm2.py` is untouched; the local checkout's
only diffs are compatibility fixes (`torch.load(weights_only=False)`, an absolutised config path, a
`deepspeed` version bound). Our pooling in `precompute_mint_embeddings.py` reproduces upstream's
`mint/helpers/extract.py` mask and mean-pooling exactly (layer 33, `~cls & ~eos & ~pad`,
`sep_chains=False`).

**Head follows upstream's own evaluation harness.** The relevant upstream reference is
`downstream/GeneralPPI/mutational-ppi`, whose README directs you to `../embeddings_mint.py` and
`../finetune_general.py` — *not* `oncoPPI/train.py`. In that harness:

- `baselines.py::encode_two` defaults to `how="subtract"`, so the **embedding difference is
  upstream's own pair-combination rule**, not a substitution of ours.
- `finetune_general.py::return_logistic_model` evaluates with `sklearn.MLPClassifier`.
- The estimator is wrapped in `GridSearchCV`, but the MLP `param_grid` is a **single point** — there
  is no hyperparameter search to replicate.
- Features are `StandardScaler`-normalized, which we match. (`PowerTransformer` is on upstream's
  regression path only.)

Our head now matches that grid: `hidden_layer_sizes=(640,)`, `activation='relu'`, `solver='adam'`,
`learning_rate='adaptive'`, `learning_rate_init=1e-3`, `max_iter=100`, `early_stopping=True`,
`validation_fraction=0.1`, `tol=1e-4`, `alpha=1e-4`.

**Deviation corrected.** Earlier runs used `hidden_layer_sizes=(64,)`, `max_iter=200` and a constant
learning rate — a 10× smaller head than MINT's authors prescribe. Corrected and all four configs
re-run for 30 seeds on `sahni_fragoza`:

| | published (64-unit) | upstream (640-unit) | Δ macro |
|---|---|---|---|
| MINT seq_diff | 0.7650 | 0.7653 | **+0.0002** |
| MINT site_diff | 0.7854 | 0.7897 | **+0.0043** |
| PPLM seq_diff | 0.7458 | 0.7485 | **+0.0027** |
| PPLM site_diff | 0.7808 | 0.7837 | **+0.0029** |

Every delta is positive and small — the shrunken head was mildly handicapping both baselines,
but not enough to affect any comparison. Per class the effect is mostly a redistribution: C1
gains (up to +0.0128 for PPLM seq_diff) while C2 slips slightly.

**One systematic effect worth stating:** the 640-unit head roughly **doubles the seed-to-seed
standard deviation in C3**, the hardest class, in all four configs (PPLM seq 0.0180 → 0.0367;
PPLM site 0.0158 → 0.0377; MINT seq 0.0171 → 0.0346; MINT site 0.0160 → 0.0327). The larger
head is less stable across resamplings even where its mean is no better. Both sets of numbers
are reported.

---

## PPLM

**This baseline is deliberately not PPLM-PPI.** The design holds the classifier head fixed and swaps
only the protein language model: MINT's pLM embeddings versus PPLM's **base** language model
embeddings. `PPLMSeqDiff`/`PPLMSiteDiff` subclass the same `CacheMLPPredictor` as
`MINTSeqDiff`/`MINTSiteDiff`; only `_extract_features` differs.

PPLM-PPI — the attention-feature classifier in upstream `pplm_ppi/model.py`, which consumes 660-dim
attention features through a 4-layer LayerNorm MLP — is **intentionally not used**. There is no
`-PPI` base model in this comparison. A reader should not interpret the PPLM row as PPLM-PPI's
performance; it measures PPLM's base embeddings under a shared head.

Embedding extraction is faithful: `_pplm_forward` builds the inter-chain mask and slices
`embed_A`/`embed_B` identically to upstream `run_pplm-ppi.py`. Upstream checkout diffs are a
circular-import fix and config path placeholders — no model change.

**Known gap.** Pairs that OOM during embedding are skipped, and the predictor then returns the class
prior for them. Unlike the MINT and ESM caches, the PPLM cache has no hit-rate audit gate.

---

## SWING

`src/evaluation/swing_gcv.py` is the in-repo, `--dataset`-parametrized port of the lab's
notebook export (the original export, `swing_cv.py`, is in `archive/dead_scripts_20260908/`;
it was unparametrized, had 22 top-level side effects and no longer imports). Hyperparameters,
in `src/evaluation/swing_common.py`, are
preserved exactly from upstream: `window_k=1`, `k=7`, `padding_score=9`; Doc2Vec `dim=128, dm=1,
alpha=0.08711, window=6, epochs=52, min_count=1`; XGBoost `n_estimators=375, max_depth=6,
learning_rate=0.08966`; `random.seed(42)`.

**Two modes, one with leakage — both reported and labelled.**
`STRINGENT_PRETRAIN=0` ("Test Pretrain") trains Doc2Vec on the entire dataset including test folds.
`STRINGENT_PRETRAIN=1` ("Blind-Test") retrains Doc2Vec per fold on training rows only and infers
test vectors. `src/analysis/export_reconstruction_tables.py` maps them to distinct display names.
**Any headline comparison uses the Blind-Test variant.**

The Doc2Vec model is not distributed: it is data-dependent and is retrained per fold for the
blind-test variant, so it is part of the model rather than a reusable artifact.

---

## SAAMBE-3D

`src/evaluation/saambe-3d.py` and `src/evaluation/utils/protseqfeature.py` are byte-identical to
upstream. The driver differs only in path constants.

**Documented model-file substitution.** In-repo `classification_v01.model` (md5 `5bb4d88b…`) is
byte-identical to `classification.model`, not to upstream's `classification_v01.model`
(md5 `1fe3e57d…`, preserved as `.bak`). Upstream's v01 file **fails to load** under xgboost 1.7.6+
(`Check failed: base_score`), which is why it was replaced. The substitution is almost certainly a
benign format fix, but it is a modification of a third-party model file and **cannot be proven
equivalent**, because the original never loads. Classification-mode predictions come from the
substitute.

---

## DDMut-PPI — excluded, with evidence

DDMut-PPI is a remote REST service; there is no local model. It is **excluded from the reported
comparisons**, and the reason is a measured server-side failure rate, not a client defect.

The in-repo client had lagged the working script by 125 lines, missing submit-retry with exponential
backoff, per-job timeout/resubmission, and `--retry-nans`. That gap has been closed — the client now
carries the full retry machinery.

Closing it does not change the conclusion. The run that produced the excluded results **already had
that machinery active** (2,774 retry/resubmit log lines), and still ended with **252 jobs timed out
after 5 resubmissions each**, an 87% NaN rate over 664 cached variants. Where jobs did complete, the
scores are sensible (305 predictions, ΔΔG range 0.007–1.812), so the service works — it simply fails
to complete most jobs for these complexes.

**Removed from the codebase (2026-09-10).** `src/evaluation/ddmutppi_cv.py` is archived to
`archive/dead_scripts_20260910/`; DDMutPPI no longer appears in `METHODS_TO_COMPARE`
(`varchamp_blind_test.py`), `roc_plots.py`'s color/baseline tables, `biclass_sf_gcv.py`'s
`SKEMPI_METHODS`, or `method_names.py`'s baseline prefixes. The zero-row VCFP arrays
(`results/varchamp_seqcnf_newvar_eval/DDMutPPI*`, orphan **O4** in
`docs/FIGURE_INVENTORY.md`) have been deleted rather than regenerated. The exclusion comments
that remain in `biclass_sf_gcv.py` and `roc_plots.py` now cite this section directly instead of
repeating the imprecise "API returns NaN for all variants" wording.

---

## MutPred2, RaSP, SPURS

**MutPred2** (`c92bf33`, clean checkout) is run externally; `src/analysis/import_mutpred2_vcfp_scores.py`
only parses its output CSV. No deviation is possible.

**RaSP** and **SPURS** are **not benchmarked**. SPURS supplies the MegaScale artifacts consumed by
`src/training/preprocess_stability_data.py` (its local diffs are pytorch-lightning API shims, no
model change). RaSP is referenced only as the provenance of the pre-MegaScale ablation checkpoint
`weights/v1_0/`.

---

## SFVCFP dataset definition change (2026-09-07)

> **Historical record.** The loaders and scripts named below no longer exist under `src/`.
> `load_sahni_fragoza_varchamp_full_pooled` and its siblings lived in `mutpred_ppi_cv.py`, which
> has since been reduced to `src/training/train_fold.py` (training loop only, no loaders, no
> CLI); data loading is now `src/utils/gcv_common.py` + `src/utils/mutpred_ppi_data.py` reading
> the canonical tables. The `supplement_*_vc1pcava.py`, `merge_vc1pcava_into_main.py` and
> `restratify_vcfp_blind_test.py` workaround scripts are in `archive/dead_scripts_20260907/`.
> The live datasets are the `*_mapped090826` tables listed in
> [`docs/REPRODUCING_ANALYSES.md`](REPRODUCING_ANALYSES.md). This section is kept because it
> explains why the SFVCFP numbers changed.

`sahni_fragoza_varchamp_full_pooled` (SFVCFP) previously mixed two identifier
namespaces. `load_sahni_fragoza_varchamp_full_pooled` normalised only the Sahni+Fragoza
source; VarChAMP1p and CAVA were merged with their raw gene-symbol+ORF ids while every other
source used UniProt:

| | rows | share | example |
|---|---|---|---|
| UniProt-keyed | 19,385 | 86.8% | `Q5TD97_O96015` |
| gene-symbol-keyed | 2,936 | 13.2% | `RAD51D_7201_CCNL1_5716` |

Downstream code that works in UniProt space — in particular the VarChAMP blind test — could
not match the gene-symbol rows and silently dropped them. A per-method
`supplement_*_vc1pcava.py` pipeline, plus `merge_vc1pcava_into_main.py` and
`restratify_vcfp_blind_test.py`, existed solely to recompute those 2,936 rows separately and
graft them back on afterwards.

The sibling loader `load_sahni_fragoza_varchamp1p_cava` had always performed the remap
correctly, using the same `gene_symbol_to_uniprot.pkl` mapping; the three lines were simply
never carried across to SFVCFP. They now are, so the dataset lives in one namespace and the
supplement/merge/restratify workaround is unnecessary.

**This changes the dataset, not just the code.** Of the 2,936 remapped rows, **1,028 turn out
to be the same complex+variant as an existing UniProt row** — duplicate measurements that the
previous code could not detect because the two names looked distinct. `_dedup_and_merge`
collapses them, so:

    n: 22,321 -> ~21,293

Because `all_vt_ids` changes, so do the cd-hit clusters, the GroupKFold splits, the C1/C2/C3
assignments and every SFVCFP result. All SFVCFP GCV runs and the VarChAMP blind test
(Fig 4, S2) are being regenerated on the corrected dataset. The previous canonical reference
is preserved under `archive/sfvcfp_reference_prefix_namespace/` so the published numbers stay
attributable.

Datasets other than SFVCFP are unaffected: `sahni` and `sahni_fragoza` never contained
gene-symbol-keyed rows.
