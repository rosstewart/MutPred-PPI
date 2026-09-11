#!/usr/bin/env python3
"""
AlphaFold3 Contact Graph Generator
Generates residue contact graphs from AlphaFold3 mmCIF structures.

Author: Ross Stewart, September 2025

Usage:
    python 01_make_contact_graphs_and_fasta.py <working_dir> <mmcif_dir> <variants_file> <n_jobs>
"""

import csv
import os
import glob
import argparse
from joblib import Parallel, delayed

from contact_graphs import (
    DEFAULT_THRESHOLD, ContactGraphStore, contact_graph_from_structure,
)
from utils import mutations


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('working_dir')
parser.add_argument('mmcif_dir')
parser.add_argument('variants_file')
parser.add_argument('n_jobs', type=int, nargs='?', default=-1)
args = parser.parse_args()

wd = args.working_dir
mmcif_dir = args.mmcif_dir
variants_file = args.variants_file
n_jobs = args.n_jobs

# setup directories
save_dir = os.path.join(wd, 'af3_graphs')
os.makedirs(wd, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

# constants
#
# The 4.5 A any-heavy-atom contact rule has ONE definition, in contact_graphs.
# It used to be re-declared here as a literal; the values agreed, but a repo
# with two copies of its own contact threshold is one edit away from building
# training graphs and inference graphs under different rules.
EDGE_DIST_THRESHOLD = DEFAULT_THRESHOLD
#
# A local `THREE_LETTER_TO_ONE` table lived here and was referenced by nothing.
# It was also a FOURTH residue policy that disagreed with the live one
# (`contact_graphs.residue_to_one`): it mapped Sec->U, Pyl->O, Asx->B, Glx->Z
# rather than folding them, and had no MSE entry at all -- so selenomethionine,
# which is common in real structures, would have been dropped. Removed
# 2026-09-10; `contact_graph_from_structure` owns residue translation.


def get_labeled_residues(variant_file):
    """Parse variant file to get mutation positions and complex IDs."""
    all_pos_indices = {}
    all_complex_ids = set()
    n_variant_partner_interactions = 0
    
    with open(variant_file, 'r') as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            id_a, variant, id_b = line.strip().split('\t')
            n_variant_partner_interactions += 1
            
            complex_id = f'{id_a}:{id_b}'
            
            # parse variant notation (e.g., V123A)
            wt_res = variant[0]
            res_idx = mutations.index(variant)  # convert to 0-based
            mt_res = variant[-1]
            
            if complex_id not in all_pos_indices:
                all_pos_indices[complex_id] = []
            
            all_pos_indices[complex_id].append(('A', res_idx, wt_res, mt_res))
            all_complex_ids.add(complex_id)
    
    return all_pos_indices, all_complex_ids, n_variant_partner_interactions


# AF3 wraps the JSON `name` field in a job prefix/suffix, e.g. a job named
# `P12345__Q67890` yields `fold_p12345__q67890_model_0.cif`. These strip that
# wrapper so the pair token can be compared directly.
_AF3_PREFIXES = ("fold_",)
_AF3_SUFFIX_RE = __import__("re").compile(r"_model(_\d+)?$|_seed(_)?\d+$")


def _strip_af3_wrapper(stem):
    for pre in _AF3_PREFIXES:
        if stem.startswith(pre):
            stem = stem[len(pre):]
    prev = None
    while prev != stem:                      # `_model_0` may follow `_seed_1`
        prev = stem
        stem = _AF3_SUFFIX_RE.sub("", stem)
    return stem


def find_mmcif_file(id_a, id_b, mmcif_dir):
    """Find the mmCIF holding both proteins. Returns `(filepath, swapped)`.

    Resolution is by FILENAME, unavoidably: this is the public 3-step pipeline,
    where the user points at their own AlphaFold3 output directory and there is
    no canonical `manifest.csv` to resolve against. (Inside the repo, structures
    are resolved content-addressed by sequence hash via
    `contact_graphs.StructureResolver`; that is strictly better and is what all
    internal code uses, but it needs a manifest this path does not have.)

    `__` IS TRIED FIRST, AND IT IS THE SEPARATOR STEP 00 EMITS.
    -----------------------------------------------------------
    This function previously tried only `{a}_{b}` and `{a}-{b}` joins, in eight
    case variants each -- and no `__` case at all. Step 00
    (`00_make_af3_json_input.py`) names every job `{id_a}__{id_b}`, precisely
    because a single `-` cannot be split back into two accessions when either is
    an isoform (`O14787-2-Q13207` is ambiguous). So a user who followed
    `docs/INFERENCE.md` step 1, folded the result, and ran step 2 got
    `FileNotFoundError` on every pair: the two halves of the same three-step
    pipeline disagreed about the separator. Fixed 2026-09-10.

    A `__` match is exact on the whole (unwrapped) stem, so isoform accessions
    round-trip unambiguously. The legacy single-separator joins are still tried
    afterwards, as a substring match with word-boundary checks, so directories
    named by older conventions keep working -- but they are only reached when
    the unambiguous form does not match.
    """
    found_files = []

    for ext in ['.cif', '.mmcif']:
        pattern = os.path.join(mmcif_dir, f'*{ext}')
        for filepath in glob.glob(pattern):
            filename = os.path.basename(filepath).replace(ext, '')

            # -- 1. the canonical `__` form: exact, isoform-safe -----------
            stem = _strip_af3_wrapper(filename)
            if "__" in stem:
                lhs, _, rhs = stem.partition("__")
                if (lhs.lower(), rhs.lower()) == (id_a.lower(), id_b.lower()):
                    found_files.append((filepath, False))
                    continue
                if (lhs.lower(), rhs.lower()) == (id_b.lower(), id_a.lower()):
                    found_files.append((filepath, True))
                    continue
                # A `__` stem that names a DIFFERENT pair is a definite
                # non-match; do not let the substring fallback below reinterpret
                # it (`A__B` contains neither `A_B` nor `A-B`, but an isoform
                # accession could still produce a spurious boundary hit).
                continue

            # -- 2. legacy single-separator joins --------------------------
            id_combinations = []
            for sep in ('_', '-'):
                a, b = id_a, id_b
                id_combinations += [
                    (f"{a}{sep}{b}", False), (f"{b}{sep}{a}", True),
                    (f"{a.lower()}{sep}{b.lower()}", False),
                    (f"{b.lower()}{sep}{a.lower()}", True),
                    (f"{a}{sep}{b.lower()}", False),
                    (f"{a.lower()}{sep}{b}", False),
                    (f"{b}{sep}{a.lower()}", True),
                    (f"{b.lower()}{sep}{a}", True),
                ]

            for id_pattern, swapped in id_combinations:
                if id_pattern in filename:
                    # verify these are the IDs and not part of a longer string,
                    # by requiring a separator or string edge on both sides
                    idx = filename.find(id_pattern)
                    valid = True
                    if idx > 0 and filename[idx-1] not in ['_', '-']:
                        valid = False
                    end_idx = idx + len(id_pattern)
                    if end_idx < len(filename) and filename[end_idx] not in ['_', '-']:
                        valid = False
                    if valid:
                        found_files.append((filepath, swapped))
                        break

    # remove duplicates
    unique_files = {}
    for filepath, swapped in found_files:
        real_path = os.path.realpath(filepath)
        if real_path not in unique_files:
            unique_files[real_path] = (filepath, swapped)

    # validate findings
    if len(unique_files) == 0:
        raise FileNotFoundError(
            f"No mmCIF file found for {id_a} and {id_b} in {mmcif_dir}. "
            f"Step 00 names AlphaFold3 jobs '{id_a}__{id_b}', so the expected "
            f"filename is '{id_a}__{id_b}.cif' (or AF3's "
            f"'fold_{id_a.lower()}__{id_b.lower()}_model_0.cif').")
    elif len(unique_files) > 1:
        raise ValueError(f"Multiple mmCIF files found for {id_a} and {id_b}: {list(unique_files.keys())}")

    return list(unique_files.values())[0]


def make_graph(complex_id, mmcif_dir, save_dir):
    """Contact graph for one complex, as a record the caller stores.

    The contact rule itself lives in `contact_graphs.contact_graph_from_structure`
    -- one definition for the whole repo, so the inference pipeline and the
    canonical graph rebuild cannot drift apart. This function only resolves WHICH
    file to read and WHICH chain the user called the interactor.

    `edge_index` is indexed over the chains in FILE order (`store_seq_a` then
    `store_seq_b`), which is what gets handed to `ContactGraphStore.put`. The
    interactor/partner sequences are recorded separately for the CSV sidecar: the
    store is keyed on sequence content and re-orients on read, so the two do not
    have to agree and no renumbering is needed here.
    """
    id_a, id_b = complex_id.split(':')

    mmcif_file, swapped = find_mmcif_file(id_a, id_b, mmcif_dir)
    built = contact_graph_from_structure(mmcif_file, EDGE_DIST_THRESHOLD)
    if built is None:
        raise ValueError(f"{mmcif_file}: not a readable two-chain structure")
    seq_a, seq_b, edge_index = built

    # `swapped` says the file lists id_b first, so id_a is the SECOND chain.
    interactor_sequence, partner_sequence = (
        (seq_b, seq_a) if swapped else (seq_a, seq_b))

    return {
        'complex_id': complex_id,
        'interactor': id_a,
        'partner': id_b,
        'interactor_sequence': interactor_sequence,
        'partner_sequence': partner_sequence,
        'store_seq_a': seq_a,
        'store_seq_b': seq_b,
        'edge_index': edge_index,
        'source': mmcif_file,
    }


def write_variant_labels(variant_indices, complex_seqs, save_dir,
                         method='interaction_loss'):
    """Write `variants.csv`: one row per (interactor, partner, mutation).

    `complex_seqs` maps `{interactor}_{partner}` -> `(seq_a, seq_b)`. It used to
    read those sequences back off disk from a `.labels_separated` file (chains
    distinguished by LETTER CASE) plus a `.num_residues_a` file holding a single
    integer -- a private serialisation of data this process already had in
    memory, in a format `utils.legacy_guard` rejects. Passing the sequences in
    removes the round trip and the two retired files with it (2026-09-10).

    Four other outputs were dropped at the same time, all written and read by
    nothing: `{a}_{b}.{method}_pos`, `{a}_{b}.labels`, `all_variants.labels`
    and `all_variants.labels_separated`. `variants.csv` is the canonical form
    and is what `generate_fasta_output` and `inference_utils` actually read.

    `mutation` is 0-BASED here, matching the ProtT5 keys the next step looks up;
    conversion to the canonical 1-based form happens once, in
    `inference_utils.write_output`.
    """
    variant_rows = []
    num_bad_variants = 0

    for complex_id in variant_indices:
        id_a, id_b = complex_id.split(':')
        key = f'{id_a}_{id_b}'
        if key not in complex_seqs:
            continue
        seq_a, seq_b = complex_seqs[key]
        pdb_seq = seq_a + seq_b
        num_residues_a = len(seq_a)

        for chain, mt_idx, wt_res, mt_res in variant_indices[complex_id]:
            if mt_idx >= num_residues_a or pdb_seq[mt_idx] != wt_res:
                print(f'  bad variant {key} {wt_res}{mt_idx}{mt_res}: '
                      f'chain={chain} num_residues_a={num_residues_a} '
                      f'found={pdb_seq[mt_idx] if mt_idx < len(pdb_seq) else "<oob>"}')
                num_bad_variants += 1
                continue
            assert chain == 'A', f'only chain A mutations are supported, got {chain}'
            variant_rows.append((id_a, id_b, f'{wt_res}{mt_idx}{mt_res}'))

    print(f'{num_bad_variants} bad variants')

    variants_index = os.path.join(save_dir, 'variants.csv')
    with open(variants_index, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['interactor', 'partner', 'mutation'])
        w.writerows(variant_rows)
    print(f'Wrote {variants_index} ({len(variant_rows)} variants)')


def generate_fasta_output(save_dir, wd):
    """Write `wt_and_vt.fasta`: every wild-type, partner and variant sequence.

    Driven by `complexes.csv` and `variants.csv`, the two tables written above.
    It previously globbed `*.interaction_loss_pos`, took the complex id from the
    FILENAME, split it on `_` to recover the two accessions, and separated the
    two chains by LETTER CASE inside a `.labels_separated` file. Every one of
    those steps is a guess: `split('_')` mis-assigns both proteins for any
    RefSeq-style id (`NP_002046_GFAP`), and case-encoding cannot represent a
    sequence that legitimately contains both cases. The tables state all of it.
    """
    complexes_path = os.path.join(save_dir, 'complexes.csv')
    variants_path = os.path.join(save_dir, 'variants.csv')

    seqs = {}                       # accession -> sequence
    with open(complexes_path) as f:
        for row in csv.DictReader(f):
            seqs.setdefault(row['interactor'], row['interactor_sequence'])
            seqs.setdefault(row['partner'], row['partner_sequence'])

    variants = []                   # (interactor, mutation_0b), de-duplicated
    seen = set()
    with open(variants_path) as f:
        for row in csv.DictReader(f):
            key = (row['interactor'], row['mutation'])
            if key not in seen:
                seen.add(key)
                variants.append(key)

    n_wt = n_vt = 0
    # Sorted so the file is byte-reproducible: dict insertion order follows
    # whatever order the structures happened to parse in, which made two runs of
    # the same input differ only by line order.
    with open(os.path.join(wd, 'wt_and_vt.fasta'), 'w') as f_out:
        for accession, seq in sorted(seqs.items()):
            f_out.write(f">{accession}\n{seq}\n")
            n_wt += 1
        for interactor, mutation in sorted(variants):
            seq = seqs.get(interactor)
            if seq is None:
                continue
            # variants.csv holds 0-BASED positions, so `position` (raw integer,
            # no base assumed) is the right accessor -- `index` would subtract
            # one from a number that is already an index.
            idx = mutations.position(mutation)
            if idx >= len(seq) or seq[idx] != mutation[0]:
                continue
            f_out.write(f">{interactor} {mutation}\n"
                        f"{seq[:idx]}{mutation[-1]}{seq[idx + 1:]}\n")
            n_vt += 1
    print(f'Wrote {n_wt} wild-type and {n_vt} variant sequences to '
          f'{os.path.join(wd, "wt_and_vt.fasta")}')


if __name__ == "__main__":
    # load variant data
    variant_indices, complex_ids, n_variant_partner_interactions = get_labeled_residues(variants_file)
    print(f'Loaded {n_variant_partner_interactions} variant-partner interactions')
    
    # generate contact graphs in parallel
    print(f'Generating {len(complex_ids)} contact graph{"s" if len(complex_ids) != 1 else ""}...')
    records = [r for r in Parallel(n_jobs=n_jobs)(
        delayed(make_graph)(complex_id, mmcif_dir, save_dir)
        for complex_id in complex_ids
    ) if r is not None]

    # One container keyed by chain sequence, plus the sidecar that maps a
    # complex_id to its two chains. Step 02 reads the sidecar and looks the graph
    # up BY SEQUENCE; it never parses a filename, which is what made isoform
    # accessions (`O43889-2-J3QKU0`) unsplittable and chain order guesswork.
    store_path = os.path.join(save_dir, 'contact_graphs.h5')
    if os.path.exists(store_path):
        os.remove(store_path)
    with ContactGraphStore(store_path, mode='w') as store:
        for r in records:
            # File order, matching `edge_index`. The store re-orients on read.
            store.put(r['store_seq_a'], r['store_seq_b'],
                      r['edge_index'], source=r['source'],
                      threshold=EDGE_DIST_THRESHOLD)
    print(f"Wrote {len(records)} graphs to {store_path}")

    index_path = os.path.join(save_dir, 'complexes.csv')
    with open(index_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['complex_id', 'interactor', 'partner',
                                          'interactor_sequence', 'partner_sequence'])
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"Wrote {index_path}")

    # Sequences stay in memory and are handed to the variant step directly.
    # They used to be round-tripped through three per-complex files
    # (`.labels`, `.labels_separated`, `.num_residues_a`) in `save_dir`.
    complex_seqs = {
        f"{r['interactor']}_{r['partner']}":
            (r['interactor_sequence'], r['partner_sequence'])
        for r in records
    }

    # generate variants.csv
    write_variant_labels(variant_indices, complex_seqs, save_dir)
    
    # generate final FASTA output
    generate_fasta_output(save_dir, wd)
