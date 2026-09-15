"""Inference-side helpers for the standalone MutPred-PPI pipeline.

Graphs come from the `ContactGraphStore` written by step 01 and are looked up by
CHAIN SEQUENCE, through the `complexes.csv` sidecar that step 01 writes
alongside it. What that replaces: a `glob` of `*.mat`, a `loadmat` per complex,
and `complex_id.split('_')` to recover the two accessions -- which silently
mis-assigns both proteins whenever an accession itself contains the separator
(isoforms, RefSeq ids). The pair identifiers are now read from their own
columns, and the composite key is CONSTRUCTED from them rather than split apart.

The store also returns the graph oriented to the requested interactor and with
self-loops already added, so the `.toarray()` / `fill_diagonal` dance each
consumer used to hand-roll is gone.
"""
import csv
from collections import Counter

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import h5py
import os
import numpy as np
import pickle
import torch
from .model_loader import get_models, model_predict
import tempfile
import joblib # for mutation diff scaler

# No sys.path bootstrap: `pip install -e .` puts src/ on the path (pyproject.toml
# exposes contact_graphs, utils, model, inference as top-level packages). The
# `utils` below is unambiguously the repo-level src/utils -- this package was
# itself called `utils` until 2026-09-10, which is what made that ambiguous; see
# inference/pipeline/__init__.py.
from contact_graphs import ContactGraphStore
from utils import mutations

# for running T5
from .prott5_loader import get_T5_model, run_T5_from_model

'''
write output
'''
def write_output(out_f, ppi_preds, rows, display=None):
    """Predictions with EXPLICIT columns: interactor, partner, mutation, score.

    `rows` is [(interactor, partner, mutation)], 1-based, straight from `variants.csv`.

    `display` maps (interactor, partner, mutation) -> the variant AS THE USER WROTE IT,
    from `variants.csv`'s `mutation_input` column. It matters whenever positions were
    supplied in a structure's own residue numbering: the sidecar holds the SEQUENTIAL
    position, so for 1H9D (RUNX1 chain starting at residue 54) reporting that back gave
    every position shifted by 53 against the input. Falls back to the sidecar position
    when the column is absent, which is right when input and sequential numbering agree.

    Both are 1-based; nothing here re-bases. `variants.csv` used to hold 0-based positions
    and this function converted, which is why the note about `to_one_based` is gone.
    """
    assert len(ppi_preds) == len(rows)
    display = display or {}
    with open(out_f, 'w') as f:
        f.write('interactor\tpartner\tmutation\tscore\n')
        for pred, (interactor, partner, mutation) in zip(ppi_preds, rows):
            shown = display.get((interactor, partner, mutation)) or mutation
            f.write(f'{interactor}\t{partner}\t{shown}\t{pred}\n')


'''
refactored function to work with on-the-go T5 embedding generation
'''
def sequence_available(key, fasta_dict):
    """True if we have a SEQUENCE for `key`.

    Named `t5_emb_exists` until 2026-09-10, which said "embedding" while testing
    a sequence dict -- the two are not the same thing at this point in the
    pipeline, and reading it as an embedding check hides why a variant is skipped.
    """
    return key in fasta_dict

# `read_h5` lived here; it is now `utils.embeddings.load_embeddings_h5`, shared
# with the variant-DB pipeline, which had a byte-for-byte equivalent copy.
from utils.embeddings import load_embeddings_h5 as read_h5


'''
generate T5 embeddings as each data point is processed to not use disk space
'''
def get_t5_emb(key, device, fasta_dict, dataset_name, t5_model, t5_vocab):
    # make temp fasta file containing only sequence of interest as well as output file
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.fasta', delete=False) as temp_fasta, \
         tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_out:
        
        temp_fasta_path = temp_fasta.name
        temp_out_path = temp_out.name

        # write only the target sequence to the temp FASTA file
        SeqIO.write(SeqRecord(Seq(fasta_dict[key]), id=key, description=""), temp_fasta_path, "fasta") 

        # execute T5. only run with visble cuda 0 (set only one visible device)
        run_T5_from_model(temp_fasta_path, temp_out_path, t5_model, t5_vocab, device)

        h5_out = read_h5(temp_out_path)
        if key not in h5_out:
            t5_emb = None
        else:
            t5_emb = h5_out[key]

    # clean up temporary files
    os.remove(temp_fasta_path)
    os.remove(temp_out_path)

    return t5_emb

'''
function to get concatenated T5 embeddings, for GAT, with missense variant accounted for
'''
def get_complex_and_vt_emb(refseq_id, partner_id, variant, INCLUDE_STABILITY, device, t5_fasta_dict, dataset_name, t5_model, t5_vocab, refseq_emb=None, vt_emb=None, partner_emb=None, scaler=None):
    # The two accessions are passed in, never recovered by splitting a composite
    # id: `complex_id.split('_')` mis-assigns both proteins for any accession
    # that contains the separator.
    mut_idx = mutations.index(variant)   # 1-based: the inference sidecars carry the same convention as every other
    # table. 0-based survives only in the variant-db/training caches, where the
    # persistent ProtT5/H5 keys are too expensive to re-key; this pipeline builds
    # its embedding H5 in a temp file and deletes it, so it had nothing to gain.
    vt_id = f'{refseq_id} {variant}'

    if refseq_emb is None:
        refseq_emb = get_t5_emb(refseq_id, device=device, fasta_dict=t5_fasta_dict, dataset_name=dataset_name, t5_model=t5_model, t5_vocab=t5_vocab)
    if vt_emb is None:
        vt_emb = get_t5_emb(vt_id, device=device, fasta_dict=t5_fasta_dict, dataset_name=dataset_name, t5_model=t5_model, t5_vocab=t5_vocab)
    if partner_emb is None:
        partner_emb = get_t5_emb(partner_id, device=device, fasta_dict=t5_fasta_dict, dataset_name=dataset_name, t5_model=t5_model, t5_vocab=t5_vocab)

    try:
        return np.concatenate([vt_emb,partner_emb]), scaler.transform(np.array(vt_emb[mut_idx] - refseq_emb[mut_idx]).reshape(1, -1)), len(refseq_emb), len(partner_emb)
        
    except Exception as e:
        # something went wrong with the variant fasta or T5 embedding size; is infrequent
        print(f'{refseq_id}_{partner_id} {variant}', e, flush=True)
        return None, None, None, None


# `get_vt_seq` lived here and was called by nothing (removed 2026-09-10). It
# duplicated `utils.mutations.apply` and DISAGREED with it in two ways: it read
# the position with `mutations.position` (raw, no base assumed) where `apply`
# treats the mutation as 1-based, and it `assert`ed on a wild-type mismatch
# instead of returning None. The house rule is stated at src/utils/mutations.py:
# a mismatch returns None and the caller must count it, because a silently
# unmutated sequence is indistinguishable from a wild-type one.

'''
function to make sequence dict from fasta file
'''
def get_dict_from_fasta(fasta_path):
    from utils.sequences import read_fasta, whole_header
    return read_fasta(fasta_path, whole_header, on_duplicate="last")

def get_vts_from_wt(save_dir):
    """{(interactor, partner): [mutation]} from `variants.csv`, 1-based.

    Reads the explicit table step 01 writes. It used to parse
    `all_variants.labels` FASTA headers of the form
    `P03372_Q14686_interaction_loss_variant_G89R`, splitting a four-part
    composite on `_interaction_loss` and then on `_` -- which breaks on any
    identifier containing the delimiter.
    """
    import csv as _csv

    path = os.path.join(save_dir, 'variants.csv')
    wt_to_vt = {}
    with open(path) as f:
        for row in _csv.DictReader(f):
            wt_to_vt.setdefault((row['interactor'], row['partner']), []).append(
                row['mutation'])
    return wt_to_vt

def read_variant_display_map(save_dir):
    """{(interactor, partner, mutation): mutation_input} from `variants.csv`.

    Empty when the table predates the `mutation_input` column, which makes
    `write_output` fall back to re-basing.
    """
    import csv as _csv

    path = os.path.join(save_dir, 'variants.csv')
    out = {}
    with open(path) as f:
        for row in _csv.DictReader(f):
            shown = row.get('mutation_input')
            if shown:
                out[(row['interactor'], row['partner'], row['mutation'])] = shown
    return out


def read_complex_index(save_dir):
    """Rows of the `complexes.csv` sidecar written by step 01.

    complex_id, interactor, partner, interactor_sequence, partner_sequence --
    the two accessions and their two chain sequences in their own columns, which
    is what makes the graph lookup a content address rather than a name match.
    """
    index_path = os.path.join(save_dir, 'complexes.csv')
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f'{index_path} not found -- run 01_make_contact_graphs_and_fasta.py '
            'first (it writes complexes.csv beside contact_graphs.h5)')
    with open(index_path) as f:
        return list(csv.DictReader(f))


'''
main inference logic
'''
def run_inference_on_dataset(device_code, dataset_name, graph_dir, t5_fasta_path, results_dir,
                             method='interaction_loss', models_dir=None, arch='current'):
    device = torch.device(device_code if torch.cuda.is_available() else 'cpu')
    print('using GPU',device)
    save_dir = graph_dir

    t5_fasta_dict = get_dict_from_fasta(t5_fasta_path) # load seqs for T5

    # ppi preds without stability are saved, so just always default to including stability to save those predictions if needed
    INCLUDE_STABILITY = 1
    BAGGED = 0

    if models_dir is None:
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'weights'))
    model_dir = os.path.abspath(models_dir)
    assert os.path.exists(model_dir), model_dir

    # `arch` selects the model GENERATION. Everything else in this function -- contact
    # graphs, ProtT5 embeddings, mutation-diff scaling -- is shared, and verified to give
    # bit-identical scores to the retired v1.0 code tree. Only the architecture and its
    # weights are actually legacy.
    if arch == 'v1.0':
        from .model_loader import load_legacy_v1_ensemble
        models = load_legacy_v1_ensemble(model_dir, device)
    elif arch == 'current':
        models = get_models(model_dir, device)
    else:
        raise ValueError(f"unknown arch {arch!r}; expected 'current' or 'v1.0'")
    scaler = joblib.load(f'{model_dir}/mutation_diff_scaler.pkl')
    print('loaded MutPred-PPI')

    t5_model, t5_vocab = get_T5_model(model_dir=None, device=device)
    print('loaded ProtT5')
    
    all_vt_ids = []
    all_wt_ids = []
    ppi_preds = []
    stability_keys = []
    prediction_count = 0

    complexes = read_complex_index(save_dir)
    store = ContactGraphStore(f'{save_dir}/contact_graphs.h5')
    wt_to_vt = get_vts_from_wt(save_dir)
    variant_display = read_variant_display_map(save_dir)
    # Named, counted reasons: a silent `continue` is how rows used to vanish
    # from a run with nothing to point at afterwards.
    skipped = Counter()
    print(len(complexes),'complexes in',save_dir,flush=True)

    for row in complexes:
        refseq_id, partner_id = row['interactor'], row['partner']
        # Kept only for log lines; nothing is looked up by it any more.
        complex_id = f'{refseq_id}_{partner_id}'

        # Oriented to this interactor, self-loops included, keyed on the two
        # chain sequences rather than on a filename.
        wt_edge_mat = store.load_dense(interactor=row['interactor_sequence'],
                                       partner=row['partner_sequence'])
        if wt_edge_mat is None:
            print(complex_id, 'has no contact graph in the store', flush=True)
            skipped['no_graph'] += 1
            continue

        if not sequence_available(refseq_id, t5_fasta_dict) or \
                not sequence_available(partner_id, t5_fasta_dict):
            print(refseq_id,'or',partner_id, 'sequence dne')
            skipped['no_wt_sequence'] += 1
            continue

        variant_f_list = wt_to_vt.get((refseq_id, partner_id), [])
        if not variant_f_list:
            skipped['no_variants_listed'] += 1
            continue
        
        # get wt embs to not compute over and over
        refseq_emb = get_t5_emb(refseq_id, device=device, fasta_dict=t5_fasta_dict, dataset_name=dataset_name, t5_model=t5_model, t5_vocab=t5_vocab)
        partner_emb = get_t5_emb(partner_id, device=device, fasta_dict=t5_fasta_dict, dataset_name=dataset_name, t5_model=t5_model, t5_vocab=t5_vocab)
        if refseq_emb is None or partner_emb is None:
            skipped['t5_failed'] += 1
            continue
        
        for vt_labels_f in variant_f_list:
            
            variant = vt_labels_f
            mut_idx = mutations.index(variant)   # 1-based sidecars; see above
            mutated_prot_vt_id = f'{refseq_id} {variant}'
            complex_and_vt = f'{complex_id} {variant}'
            
            if not sequence_available(mutated_prot_vt_id, t5_fasta_dict):
                print(mutated_prot_vt_id,'sequence does not exist')
                skipped['no_variant_sequence'] += 1
                continue


            print(f'{complex_and_vt} generating T5 embeddings... ',end='',flush=True)
            
            try:
                prott5_embedding, mutation_emb_diff, mut_seq_len, partner_seq_len = get_complex_and_vt_emb(refseq_id, partner_id, variant, INCLUDE_STABILITY=0, device=device, t5_fasta_dict=t5_fasta_dict, dataset_name=dataset_name, t5_model=t5_model, t5_vocab=t5_vocab, refseq_emb=refseq_emb, partner_emb=partner_emb, scaler=scaler)
                if prott5_embedding is None:
                    print(f'{complex_and_vt} prott5 embedding is none',flush=True)
                    skipped['no_embedding'] += 1
                    continue

                assert mut_seq_len + partner_seq_len == prott5_embedding.shape[0]

                edge_mat = wt_edge_mat   # dense, self-loops already added
                vt_id = complex_and_vt

                # for PPI predictor
                x = prott5_embedding

                if edge_mat.shape[0] != x.shape[0]:
                    print(f'{complex_and_vt} graph has {edge_mat.shape[0]} nodes '
                          f'for {x.shape[0]} embedded residues',flush=True)
                    skipped['graph_embedding_length_mismatch'] += 1
                    continue
    
                '''
                run predictions
                '''
                print(f'running model predictions... ',end='',flush=True)
                pred = model_predict(x, edge_mat, models=models, mutation_idx=mut_idx, mutation_site_diff=mutation_emb_diff, device=device)
                
                
                all_vt_ids.append((refseq_id, partner_id, variant))
                all_wt_ids.append(complex_id)
                ppi_preds.append(float(pred))
                
                print(f'done',flush=True)
                prediction_count += 1

                # save results along the way for large requests
                if prediction_count % 2000 == 1000:
                    write_output(f'{results_dir}/MutPred-PPI_preds.tsv', ppi_preds, all_vt_ids,
                                display=variant_display)

            
            except Exception as e:
                print()
                print(f'error {e}',flush=True)
                skipped['error'] += 1

    store.close()
    for reason, count in sorted(skipped.items()):
        print(f'  skipped, {reason}: {count}', flush=True)

    # save predictions
    write_output(f'{results_dir}/MutPred-PPI_preds.tsv', ppi_preds, all_vt_ids,
                                display=variant_display)
    print(f'\nWrote {len(ppi_preds)} prediction{"s" if len(ppi_preds) != 1 else ""} to {results_dir}/MutPred-PPI_preds.tsv', end='\n\n')

    


