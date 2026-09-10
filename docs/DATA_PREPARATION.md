# Data preparation

The ordered chain that turns published source files into the tables, structures
and graphs every figure script reads. Each stage names its inputs, its outputs
and the script that performs it, so a clone can be brought to a runnable state
without guessing which artifact came from where.

Everything below writes under `datasets/`, which is **not** in git (see
[SETUP.md](SETUP.md)). Paths resolve through `src/paths.py`; nothing in the
pipeline hardcodes an absolute path.

```
  source files                notebooks/map_ppi_datasets_090826.py
  (published + restricted)  ────────────────────────────────────►  datasets/source_mapping/
                                                                          │
                             src/data_processing/training_sets/            │
                             prepare_gcv_tables.py                         ▼
                           ◄────────────────────────────────────  datasets/training_eval/
                                                                    *_rows.csv.gz
                                                                    *_splits.csv.gz
                                                                    sequences.csv.gz
                                                                          │
  AlphaFold3 predictions                                                  ▼
  ──────────────────────►  canonicalize_structures.py  ──►  datasets/af3_structures_canonical/
         ▲                                                          │        manifest.csv
         │                                                          ▼
         │                 rebuild_graphs_from_structures.py  ──►  contact_graphs.h5
         │                                                          │
         └───── fold externally ◄── prepare_af3_inputs.py ◄─────────┘
```

## Stage 0 — source data

Two tiers, both gitignored.

| tier | directory | contents |
|---|---|---|
| published | `datasets/source_data/` | `sahni_wt_and_mt_y2h_scores.csv`, `fragoza_cosmic.csv`, `fragoza_exac.csv`, `fragoza_hgmd.csv` |
| restricted | `datasets/source_data_restricted/` | four VarChAMP/IGVF files |

The published files come from the supplementary material of the two source
papers — see [DATA_SOURCES.md](DATA_SOURCES.md) for citations and download
locations. The VarChAMP/IGVF files were unpublished consortium data at time of
release and **cannot be redistributed**; they are absent from Zenodo, and every
stage below degrades cleanly without them (the `sahni_only`, `fragoza_only` and
`sahni_fragoza` datasets are buildable from the published files alone).

## Stage 1 — mapping

```bash
# as a notebook (recommended -- it emits QC artifacts meant to be read)
jupytext --to ipynb notebooks/map_ppi_datasets_090826.py && jupyter lab
# or straight through
conda run -n ppi python notebooks/map_ppi_datasets_090826.py
```

Resolves every source identifier (accession, RefSeq, Entrez GeneID, gene symbol,
hORFeome ORF) to a reviewed human UniProt accession, validates each mutation
against the resolved sequence, and writes:

| output | contents |
|---|---|
| `datasets/source_mapping/datasets/` | the five mapped dataset CSVs, plus narrower VarChAMP slices |
| `datasets/source_mapping/master_data/` | the complete un-deduplicated record, one row per observation |
| `datasets/source_mapping/intermediate_files/` | 11 QC/audit CSVs — why rows were dropped, remapped, collapsed |
| `datasets/source_mapping/cache/` | UniProt REST responses, so a re-run is offline |

**Carry the cache.** Without it the notebook queries `rest.uniprot.org` live, and
UniProt's answers change over time — which means the mapping is only exactly
reproducible with the cached responses in place. It is 20 MB.

### Ambiguous Entrez GeneIDs

A few NCBI GeneIDs cross-reference more than one *reviewed human* Swiss-Prot
entry. This is not a filter failure — the Entrez route requests
`to_db=UniProtKB-Swiss-Prot` (reviewed-only, server-side) and applies a hard
human filter client-side. UniProt genuinely maps them to several entries.

Two cases, treated differently:

- **Sequence-identical paralog families** (`122183` PRR20A–E, `441521` CT45A5/6/7).
  Every candidate carries the same sequence, so the choice cannot matter. Passes.
- **Genuinely different products under one GeneID** (`51207` DUSP13, `83871`
  RAB34, `9465` AKAP7). Here a wrong pick substitutes an unrelated protein —
  DUSP13A and DUSP13B share **3.7% identity**. These are pinned explicitly in
  `ENTREZ_AMBIGUOUS_PINS` with the reason for each, and a *new* divergent GeneID
  that is not pinned raises rather than being resolved by a length heuristic.

## Stage 2 — GCV row and split tables

```bash
conda run -n ppi python src/data_processing/training_sets/prepare_gcv_tables.py \
    --dataset all --n-seeds 30
```

Consumes the mapped CSVs and writes `datasets/training_eval/`:
`sequences.csv.gz`, and per dataset `{name}_rows.csv.gz` + `{name}_splits.csv.gz`.

`row_index` is the CSV's natural order and is never renumbered — it is the join
key between the two tables and everything downstream. Clustering is cd-hit at
50% identity **on the full concatenated complex sequence**, which is the
published convention and must not change. Folds are `GroupKFold(10, shuffle=True,
random_state=seed)` over those clusters; `test_class` is C1/C2/C3 by whether each
protein of a test pair was seen in that fold's training set.

Use `--out` to write to a scratch directory and diff against the live tables
without overwriting them.

Expected row counts: `sahni_only` 1,595 · `fragoza_only` 4,729 ·
`sahni_fragoza` 6,219 · `varchamp_all` 17,376 ·
`sahni_fragoza_varchamp_all` 23,320.

## Stage 3 — structures

AlphaFold3 predictions arrive as per-job directories. Canonicalise them into one
content-addressed tree:

```bash
conda run -n ppi python src/data_processing/canonicalize_structures.py \
    --structures datasets/af3_structures --out datasets/af3_structures_canonical
```

Files are named `{ACC_LO}__{ACC_HI}.cif.gz`. The double underscore is load-bearing:
no UniProt accession contains `_`, so isoform accessions like `O14787-2` survive
a round trip, which a single `-` join cannot (`O14787-2-Q13207` is ambiguous).
**Chain orientation is resolved at load time from the manifest, never from the
filename.**

## Stage 4 — contact graphs

```bash
conda run -n ppi python src/data_processing/rebuild_graphs_from_structures.py \
    --structures datasets/af3_structures_canonical \
    --out datasets/training_eval/contact_graphs.h5
```

A 4.5 Å any-heavy-atom contact rule (`contact_graphs.DEFAULT_THRESHOLD`, the one
definition). The store is keyed on the sorted pair of `sha256(sequence)[:16]`
hashes, so it is independent of filenames and of chain order.

**Rebuild this whenever the structure set grows.** A store with fewer graphs than
the manifest has rows is stale, and every newly-folded pair silently scores NaN.

## Stage 5 — AlphaFold3 inputs for what is still missing

```bash
conda run -n ppi python src/data_processing/prepare_af3_inputs.py
```

A standing step, not an anomaly report: it always runs, reports coverage, and
exits cleanly when nothing is missing.

- **required** = the union of unordered `(interactor, partner)` pairs across all
  five canonical datasets. Not the pooled table alone: pooling drops rows whose
  label conflicts across sources, so a pair can be absent there while present and
  unconflicted in a smaller dataset.
- **present** = any pair in the manifest, in *either* chain order.

It writes `missing_pairs.csv` (the audit record) and one AF3 JSON per pair.
JSON emission is delegated to `src/inference/00_make_af3_json_input.py` — the one
JSON writer — so the naming, dialect and residue normalisation cannot drift.

Fold the emitted JSONs on a GPU machine, then return to **stage 3**.

## Order

Stages 1 → 2 → 3 → 4, with 5 feeding back into 3. Stage 2 must precede stage 5,
since the required-pair set is derived from the tables stage 2 writes.
`notebooks/reproduce_all_figures.py` runs stages 2–5 in this order and caches on
output existence.
