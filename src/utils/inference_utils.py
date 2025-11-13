from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import h5py
import os, sys
import subprocess
import numpy as np
import pickle
from scipy.io import loadmat
import scipy.sparse as sp
import glob
import h5py
import pickle
import torch
from .model_loader import get_models, model_predict
import random
import tempfile
import joblib # for mutation diff scaler

# for running T5
from .prott5_loader import get_T5_model, run_T5_from_model

'''
write output
'''
def write_output(out_f, ppi_preds, vt_ids):
    assert len(ppi_preds) == len(vt_ids)
    with open(out_f,'w') as f:
        f.write('complex_id\tvariant\tscore\n')
        for i,pred in enumerate(ppi_preds):
            wt_id, variant = vt_ids[i].split(' ')
            variant = variant[0] + str(int(variant[1:-1])+1) + variant[-1]
            f.write(f'{wt_id}\t{variant}\t{pred}\n')


'''
refactored function to work with on-the-go T5 embedding generation
'''
def t5_emb_exists(key, fasta_dict):
    return key in fasta_dict

'''
read T5 embedding dict from .h5 file output
'''
def read_h5(in_f):
    t5_embeddings = {}
    with h5py.File(in_f, 'r') as f:
        # list all keys (groups/datasets)
        for variant_id in list(f.keys()):
            dataset = f[variant_id]
            # convert dataset to a numpy array
            t5_embeddings[variant_id] = dataset[:]
    return t5_embeddings


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
def get_complex_and_vt_emb(complex_and_vt, INCLUDE_STABILITY, device, t5_fasta_dict, dataset_name, t5_model, t5_vocab, refseq_emb=None, vt_emb=None, partner_emb=None, scaler=None):
    complex_id, variant = complex_and_vt.split(' ')
    mut_idx = int(variant[1:-1]) # already made zero-based
    refseq_id, partner_id = complex_id.split('_')
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
        print(complex_and_vt, e, flush=True)
        return None, None, None, None


'''
overengineered function to mutate a wild-type sequence given a missense variant
'''
def get_vt_seq(wt_seq, variant):
    missense_idx = int(variant[1:-1]) # assumed to be zero-based
    assert wt_seq[missense_idx] == variant[0]
    
    vt_seq = list(wt_seq)
    
    # synonymous variant. bad entry in dataset
    if vt_seq[missense_idx] == variant[-1]:
        print(f'{variant} skipping')
        return None
        
    vt_seq[missense_idx] = variant[-1]
    vt_seq = ''.join(vt_seq)

    return vt_seq


'''
function to make sequence dict from fasta file
'''
def get_dict_from_fasta(fasta_path):
    return {record.description.strip(): str(record.seq).strip() for record in SeqIO.parse(fasta_path, "fasta")}

def get_vts_from_wt(variant_labels_f):
    wt_to_vt = {}
    with open(variant_labels_f,'r') as f:
        pdb_id, variant = None, None
        for line in f:
            if line[0] == '>':
                # e.g. P03372_Q14686_interaction_loss_variant_G89R
                filename_key = line[1:].strip()
                pdb_id = filename_key.split('_interaction_loss')[0]
                if pdb_id not in wt_to_vt:
                    wt_to_vt[pdb_id] = []
                variant = filename_key.split('_')[-1]
                wt_to_vt[pdb_id].append(variant)
    return wt_to_vt

'''
main inference logic
'''
def run_inference_on_dataset(device_code, dataset_name, graph_dir, t5_fasta_path, results_dir, method='interaction_loss'):
    device = torch.device(device_code if torch.cuda.is_available() else 'cpu')
    print('using GPU',device)
    save_dir = graph_dir
    
    t5_fasta_dict = get_dict_from_fasta(t5_fasta_path) # load seqs for T5
    
    # ppi preds without stability are saved, so just always default to including stability to save those predictions if needed
    INCLUDE_STABILITY = 1
    BAGGED = 0
    
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', 'models'))
    assert os.path.exists(model_dir), model_dir

    models = get_models(model_dir, device)
    scaler = joblib.load(f'{model_dir}/mutation_diff_scaler.pkl')
    print('loaded MutPred-PPI')

    t5_model, t5_vocab = get_T5_model(model_dir=None, device=device)
    print('loaded ProtT5')
    
    all_vt_ids = []
    all_wt_ids = []
    ppi_preds = []
    stability_keys = []
    prediction_count = 0

    wt_f_mats = glob.glob(f'{save_dir}/*.mat')
    print(len(wt_f_mats),'complexes in',save_dir,flush=True)
    
    for wt_idx,wt_f_mat in enumerate(wt_f_mats):
            
        complex_id = wt_f_mat.split('/')[-1].split('.')[0]
        refseq_id = '_'.join(complex_id.split('_')[:1]) # this is just the first uniprot id; the name carried over from training
        partner_id = '_'.join(complex_id.split('_')[1:])
    
        data = loadmat(wt_f_mat)
        wt_edge_mat = sp.csr_matrix(data['G'])
        wt_seq = ''.join(data['L'])
    
        if not t5_emb_exists(refseq_id, t5_fasta_dict) or not t5_emb_exists(partner_id, t5_fasta_dict):
            print(refseq_id,'or',partner_id, 'sequence dne')
            continue
        
        wt_to_vt = get_vts_from_wt(f'{save_dir}/all_variants.labels')
        variant_f_list = wt_to_vt[complex_id]
        
        # get wt embs to not compute over and over
        refseq_emb = get_t5_emb(refseq_id, device=device, fasta_dict=t5_fasta_dict, dataset_name=dataset_name, t5_model=t5_model, t5_vocab=t5_vocab)
        partner_emb = get_t5_emb(partner_id, device=device, fasta_dict=t5_fasta_dict, dataset_name=dataset_name, t5_model=t5_model, t5_vocab=t5_vocab)
        if refseq_emb is None or partner_emb is None:
            continue
        
        for vt_labels_f in variant_f_list:
            
            variant = vt_labels_f
            mut_idx = int(variant[1:-1]) # already made zero-based
            mutated_prot_vt_id = f'{refseq_id} {variant}'
            complex_and_vt = f'{complex_id} {variant}'
            
            if not t5_emb_exists(mutated_prot_vt_id, t5_fasta_dict):
                print(mutated_prot_vt_id,'sequence does not exist')
                continue


            print(f'{complex_and_vt} generating T5 embeddings... ',end='',flush=True)
            
            try:
                prott5_embedding, mutation_emb_diff, mut_seq_len, partner_seq_len = get_complex_and_vt_emb(complex_and_vt, INCLUDE_STABILITY=0, device=device, t5_fasta_dict=t5_fasta_dict, dataset_name=dataset_name, t5_model=t5_model, t5_vocab=t5_vocab, refseq_emb=refseq_emb, partner_emb=partner_emb, scaler=scaler)
                if prott5_embedding is None:
                    print(f'{complex_and_vt} prott5 embedding is none',flush=True)
                    continue

                assert mut_seq_len + partner_seq_len == prott5_embedding.shape[0]
                
                # add chain encoding as column to embedding
                labels = np.concatenate([np.zeros(mut_seq_len), np.ones(partner_seq_len)])
                
                edge_mat = wt_edge_mat
                vt_id = complex_and_vt
                seq_length = (mut_seq_len, partner_seq_len)

                # for PPI predictor
                x = prott5_embedding
            
                try:
                    # if edge mat is sparse
                    edge_mat = edge_mat.toarray()
                except Exception as e1:
                    pass # already made as dense
                np.fill_diagonal(edge_mat,1)

                assert edge_mat.shape[0] == edge_mat.shape[1] and edge_mat.shape[0] == x.shape[0]
    
                '''
                run predictions
                '''
                print(f'running model predictions... ',end='',flush=True)
                pred = model_predict(x, edge_mat, models=models, mutation_idx=mut_idx, mutation_site_diff=mutation_emb_diff, device=device)
                
                
                all_vt_ids.append(vt_id)
                all_wt_ids.append(complex_id)
                ppi_preds.append(float(pred))
                
                print(f'done',flush=True)
                prediction_count += 1

                # save results along the way for large requests
                if prediction_count % 2000 == 1000:
                    write_output(f'{results_dir}/MutPred-PPI_preds.tsv', ppi_preds, all_vt_ids)

            
            except Exception as e:
                print()
                print(f'error {e}',flush=True)
                print(e,flush=True)
                # sys.exit(1)
    
    # save predictions
    write_output(f'{results_dir}/MutPred-PPI_preds.tsv', ppi_preds, all_vt_ids)
    print(f'\nWrote {len(ppi_preds)} prediction{"s" if len(ppi_preds) != 1 else ""} to {results_dir}/MutPred-PPI_preds.tsv', end='\n\n')

    


