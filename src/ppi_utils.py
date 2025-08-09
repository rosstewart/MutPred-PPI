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
from gnn_models import GAT_pool, GAT_residue, GAT_RaSP, GAT_mut_processor, get_ppi_models, get_stability_models, get_scaledstability_finetune_models, ppi_bagged_predict, RASP_bagged_predict, ppi_stability_finetune_predict
import random
import tempfile
import joblib # for mutation diff scaler

# for running T5
from prott5_embedder_loaded import get_T5_model, run_T5_from_model


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
        # List all keys (groups/datasets)
        for variant_id in list(f.keys()):
            dataset = f[variant_id]
            # Convert dataset to a numpy array
            t5_embeddings[variant_id] = dataset[:]
    return t5_embeddings


'''
generate T5 embeddings as each data point is processed to save disk space
'''
# def get_t5_emb(key, gpu, fasta_dict, dataset_name, t5_model, t5_vocab):
    
#     device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    
#     # print(fasta_dict)
#     t5_save_dir=f'/data/ross/ppi_lossgain/interaction_loss/t5_embs/{dataset_name}_temp'
#     os.makedirs(t5_save_dir,exist_ok=True)
    
    
#     temp_fasta_path = f'{t5_save_dir}/temp.fasta'
#     SeqIO.write(SeqRecord(Seq(fasta_dict[key]), id=key, description=""), temp_fasta_path, "fasta") 
    
    
#     temp_out_f = f'{t5_save_dir}/temp_t5.h5'

#     # run preloaded
#     run_T5_from_model(temp_fasta_path, temp_out_f, t5_model, t5_vocab, device)
#     # subprocess.run(f"CUDA_VISIBLE_DEVICES={gpu} python ../../prott5/prott5_embedder_cuda0.py --input {temp_fasta_path} --output {temp_out_f}", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # send output to /dev/null unless there is an error
   
#     t5_emb = read_h5(temp_out_f)[key]

#     # clean up temporary files
#     os.remove(temp_fasta_path)
#     os.remove(temp_out_f)

    # return t5_emb

def get_t5_emb(key, gpu, fasta_dict, dataset_name, t5_model, t5_vocab):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    # make temp fasta file containing only sequence of interest as well as output file
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.fasta', delete=False) as temp_fasta, \
         tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_out:
        
        temp_fasta_path = temp_fasta.name
        temp_out_path = temp_out.name

        # Write only the target sequence to the temp FASTA file
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
function to get concatenated T5 embeddings, for GAT, with missense variant accounted for.
optionally return mutated protein embedding difference for stability predictor
'''
def get_complex_and_vt_emb(complex_and_vt, INCLUDE_STABILITY, gpu, t5_fasta_dict, dataset_name, t5_model, t5_vocab, refseq_emb=None, vt_emb=None, partner_emb=None, seq_confirmed_code='', scaler=None):
    complex_id, variant = complex_and_vt.split(' ')
    mut_idx = int(variant[1:-1]) # already made zero-based
    refseq_id, partner_id = complex_id.split('_')
    vt_id = f'{refseq_id} {variant}'

    if refseq_emb is None:
        refseq_emb = get_t5_emb(refseq_id, gpu=gpu, fasta_dict=t5_fasta_dict, dataset_name=dataset_name, t5_model=t5_model, t5_vocab=t5_vocab)
    if vt_emb is None:
        vt_emb = get_t5_emb(vt_id, gpu=gpu, fasta_dict=t5_fasta_dict, dataset_name=dataset_name, t5_model=t5_model, t5_vocab=t5_vocab)
    if partner_emb is None:
        partner_emb = get_t5_emb(partner_id, gpu=gpu, fasta_dict=t5_fasta_dict, dataset_name=dataset_name, t5_model=t5_model, t5_vocab=t5_vocab)

    try:

        if seq_confirmed_code == '_scaledstability_finetune':
            return np.concatenate([vt_emb,partner_emb]), scaler.transform(np.array(vt_emb[mut_idx] - refseq_emb[mut_idx]).reshape(1, -1)), len(refseq_emb), len(partner_emb)
        
        # subtract wt from vt at mutated position only. use wt for rest of it
        vt_complex_emb = refseq_emb.copy()
        vt_complex_emb[mut_idx] = vt_emb[mut_idx] - refseq_emb[mut_idx]
        vt_complex_emb = np.concatenate([vt_complex_emb,partner_emb])
    
        if not INCLUDE_STABILITY:
            return vt_complex_emb, len(refseq_emb), len(partner_emb)
        else:
            # subtract all of wt from vt, use original embedding not modified one
            mutated_prot_emb_diff = vt_emb - refseq_emb
            
            return vt_complex_emb, mutated_prot_emb_diff, len(refseq_emb), len(partner_emb)
    except Exception as e:
        # something went wrong with the variant fasta or T5 embedding size; is infrequent
        print(complex_and_vt, e, flush=True)
        return None, None, None, None



'''
function to get shared training proteins and those with dirbind information in BioGRID
'''
def get_filters(train_ids_path='../../ppi_lossgain/combined_sahni_fragoza_varchamp1p_cava_seq_confirmed_train_uniprot_ids.pkl',
                biogrid_dict_path='/data/ross/ppi_lossgain/interaction_loss/biogrid/biogrid_dirbind_uniprot_to_interactors.pkl'):
    
    # discard shared proteins in training set
    with open(train_ids_path,'rb') as f:
        sahni_train_uniprot_ids = pickle.load(f)
    
    # dirbind only filter
    with open(biogrid_dict_path,'rb') as f:
        biogrid_uniprot_to_interactors = pickle.load(f)

    return sahni_train_uniprot_ids, biogrid_uniprot_to_interactors

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
if i used standard labels (.pos, .neg files) for the dataset, this extracts that information.
otherwise all variants are stored in a .pos file
'''
def get_site_label(variant, pos_f, neg_f, unlabeled_f):
    global_label = None
    
    # sanity check to make sure site exists and there are no duplicates
    site_f = pos_f
    seen_site = False
    if os.path.exists(pos_f):
        with open(site_f,'r') as f:
            for line in f:
                if line.strip().split('\t')[-1] == variant:
                    global_label = 1
                    assert not seen_site
                    seen_site = True
    if not seen_site:
        site_f = neg_f
        if os.path.exists(neg_f):
            with open(site_f,'r') as f:
                for line in f:
                    if line.strip().split('\t')[-1] == variant:
                        global_label = 0
                        assert not seen_site
                        seen_site = True
        if not seen_site:
            site_f = unlabeled_f
            with open(site_f,'r') as f:
                for line in f:
                    if line.strip().split('\t')[-1] == variant:
                        global_label = -1 # VUS for clinvar
                        seen_site = True
            if not seen_site:
                return None

    return global_label

'''
function to convert keys of variant-to-label dictionary from one-based to zero-based mutation indices
'''
def convert_one_to_zero_based_variant_dict(one_based_variant_dict):
    return {f"{key.split(' ')[0]} {key.split(' ')[1][0]}{int(key.split(' ')[1][1:-1])-1}{key.split(' ')[1][-1]}": val for key,val in one_based_variant_dict.items()}


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
def run_inference_on_dataset(gpu, dataset_name, graph_dir, t5_fasta_path, results_dir, method='interaction_loss', percent_keep=1.0, seq_confirmed_code=''):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    print(device)
    save_dir = graph_dir
    
    t5_fasta_dict = get_dict_from_fasta(t5_fasta_path) # load seqs for T5
    
    # RESIDUE_MODEL = 1
    # BOTH_MODELS = 1
    # COMPLEX_MODEL = 0
    BATCH_SIZE = 16

    # ppi preds without stability are saved, so just always default to including stability to save those predictions if needed
    INCLUDE_STABILITY = 1
    BAGGED = 0
    
    model_dir = f"/data/ross/gnn/ppi_interaction_loss{'/bagged' if BAGGED else ''}"
    assert os.path.exists(model_dir)

    if seq_confirmed_code == '_scaledstability_finetune':
        models = get_scaledstability_finetune_models(model_dir, device)
        scaler = joblib.load('/data/ross/gnn/jose_2016_lossgain_models/mutation_diff_scaler.pkl')
    else:
        residue_models, pool_models = get_ppi_models(model_dir, device, seq_confirmed_code)
        if INCLUDE_STABILITY:
            rasp_models = get_stability_models(device)
    
    t5_model, t5_vocab = get_T5_model(model_dir=None, device=device)
    # with open('/data/ross/ppi_lossgain/interaction_loss/autism_dirbind_subset_interaction_loss_wt_and_vt_t5.pkl','rb') as f:
    #     t5_model = pickle.load(f)
    
    sahni_train_uniprot_ids, biogrid_uniprot_to_interactors = get_filters()

    # different processing due to extremely large amount of files
    GNOMAD_FLAG = 0
    if 'gnomad' in t5_fasta_path:
        GNOMAD_FLAG = 1
        wt_to_vt = get_vts_from_wt(f'{save_dir}/all_variants.labels')
    
    global_labels = []
    all_vt_ids = []
    all_wt_ids = []
    ppi_preds = []
    all_stability_preds = []
    stability_keys = []
    prediction_count = 0

    
    wt_f_mats = glob.glob(f'{save_dir}/*.mat')
    print(len(wt_f_mats),'complexes in',save_dir,flush=True)
    
    for wt_idx,wt_f_mat in enumerate(wt_f_mats):
            
        complex_id = wt_f_mat.split('/')[-1].split('.')[0]
        refseq_id = '_'.join(complex_id.split('_')[:1]) # this is just the first uniprot id; the name carried over from training
        partner_id = '_'.join(complex_id.split('_')[1:])
    
        data = loadmat(wt_f_mat)
        wt_edge_mat = sp.csr_matrix(data['G'])#.toarray() save memory
        wt_seq = ''.join(data['L'])
    
        if not t5_emb_exists(refseq_id, t5_fasta_dict) or not t5_emb_exists(partner_id, t5_fasta_dict):
            # print(refseq_id,'or',partner_id, 'sequence dne')
            continue
        
        # skip shared proteins in training set
        if refseq_id in sahni_train_uniprot_ids or partner_id in sahni_train_uniprot_ids:
            continue

        # skip complexes that don't have direct binding evidence
        if refseq_id not in biogrid_uniprot_to_interactors or partner_id not in biogrid_uniprot_to_interactors or partner_id not in biogrid_uniprot_to_interactors[refseq_id] or refseq_id not in biogrid_uniprot_to_interactors[partner_id]:
            # print('not in biogrid')
            continue

        if not GNOMAD_FLAG:
            variant_f_list = glob.glob(f'{save_dir}/{complex_id}*{method}_variant*.labels')
        else:
            variant_f_list = wt_to_vt[complex_id]
            # print(f'{complex_id} {len(variant_f_list)} variants loaded')

        # get refseq_emb to not compute over and over
        refseq_emb = get_t5_emb(refseq_id, gpu=gpu, fasta_dict=t5_fasta_dict, dataset_name=dataset_name, t5_model=t5_model, t5_vocab=t5_vocab)
        if refseq_emb is None:
            continue
        
        for vt_labels_f in variant_f_list:
            
            variant = vt_labels_f.split('variant_')[-1].split('.')[0] if not GNOMAD_FLAG else vt_labels_f
            mut_idx = int(variant[1:-1]) # already made zero-based
            mutated_prot_vt_id = f'{refseq_id} {variant}'
            complex_and_vt = f'{complex_id} {variant}'
            
            if not t5_emb_exists(mutated_prot_vt_id, t5_fasta_dict):
                print(mutated_prot_vt_id,'sequence does not exist')
                continue

            
            if percent_keep != 1.0:
                if random.random() >= percent_keep: # skip x% of variants for time, not skipping any wts though
                    print(f'skipping {complex_and_vt} to save time... ',flush=True)
                    continue

            print(f'{complex_and_vt} generating T5 embeddings... ',end='',flush=True)
            
            try:
                if seq_confirmed_code == '_scaledstability_finetune':
                    prott5_embedding, mutation_emb_diff, mut_seq_len, partner_seq_len = get_complex_and_vt_emb(complex_and_vt, INCLUDE_STABILITY=0, gpu=gpu, t5_fasta_dict=t5_fasta_dict, dataset_name=dataset_name, t5_model=t5_model, t5_vocab=t5_vocab, refseq_emb=refseq_emb, seq_confirmed_code=seq_confirmed_code, scaler=scaler)
                else:
                    if not INCLUDE_STABILITY:
                        prott5_embedding, mut_seq_len, partner_seq_len = get_complex_and_vt_emb(complex_and_vt, INCLUDE_STABILITY=INCLUDE_STABILITY, gpu=gpu, t5_fasta_dict=t5_fasta_dict, dataset_name=dataset_name, t5_model=t5_model, t5_vocab=t5_vocab)
                    else:
                        prott5_embedding, mutated_prot_emb_diff, mut_seq_len, partner_seq_len = get_complex_and_vt_emb(complex_and_vt, INCLUDE_STABILITY=INCLUDE_STABILITY, gpu=gpu, t5_fasta_dict=t5_fasta_dict, dataset_name=dataset_name, t5_model=t5_model, t5_vocab=t5_vocab)
                    # print(prott5_embedding, mutated_prot_emb_diff, mut_seq_len, partner_seq_len)
                if prott5_embedding is None:
                    print(f'{complex_and_vt} prott5 embedding is none',flush=True)
                    continue

                assert mut_seq_len + partner_seq_len == prott5_embedding.shape[0]
                
                # add chain encoding as column to embedding
                labels = np.concatenate([np.zeros(mut_seq_len), np.ones(partner_seq_len)])
                if seq_confirmed_code != '_scaledstability_finetune':
                    prott5_embedding = np.concatenate([labels[:, None], prott5_embedding], axis=1)
                
                edge_mat = wt_edge_mat
                vt_id = complex_and_vt
                seq_length = (mut_seq_len, partner_seq_len)
                # get label. clinvar has labels sored in this format, but others get later
                wt_id, variant = vt_id.split()
                mutation_idx = int(variant[1:-1])
                pos_f = f'{graph_dir}/{wt_id}.{method}_pos'
                neg_f = f'{graph_dir}/{wt_id}.{method}_neg'
                unlabeled_f = f'{graph_dir}/{wt_id}.{method}_unlabeled'
                
                # print(pos_f)
                
                assert os.path.exists(pos_f) or os.path.exists(neg_f) or os.path.exists(unlabeled_f)
                global_label = get_site_label(variant, pos_f, neg_f, unlabeled_f)
                if global_label is None:
                    print(f'{complex_and_vt} global label is none',flush=True)
                    continue
        
                # assert global_label is not None
        
                # for PPI predictors
                x = prott5_embedding
                
                key = vt_id
                # already made zero-based. idk why i do this like 3 times
                variant_res_idx = int(key.split(' ')[-1][1:-1]) 
            
                try:
                    # if edge mat is sparse
                    edge_mat = edge_mat.toarray()
                except Exception as e1:
                    pass # already made as dense
                np.fill_diagonal(edge_mat,1)

                assert edge_mat.shape[0] == edge_mat.shape[1] and edge_mat.shape[0] == x.shape[0]
                
                # print(complex_and_vt, x, edge_mat, pool_models, residue_models)
                # raise Exception
    
                '''
                run predictions
                '''
                print(f'running model predictions... ',end='',flush=True)
                if seq_confirmed_code == '_scaledstability_finetune':
                    pred = ppi_stability_finetune_predict(x, edge_mat, models=models, mutation_idx=variant_res_idx, mutation_site_diff=mutation_emb_diff, device=device)
                else:
                    pred_pool = ppi_bagged_predict(x, edge_mat, models=pool_models, RESIDUE_MODEL=0, variant_res_idx=None, device=device)
                    if pred_pool is None:
                        print('pred_pool failed',flush=True)
                        continue
                    pred_residue = ppi_bagged_predict(x, edge_mat, models=residue_models, RESIDUE_MODEL=1, variant_res_idx=variant_res_idx, device=device)
                    if pred_residue is None:
                        print('pred_residue failed',flush=True)
                        continue
            
                    pred = np.mean((pred_pool, pred_residue))
                    
                    if INCLUDE_STABILITY:
                
                        mutated_seq_length = seq_length[0]
                        Stability_pred = RASP_bagged_predict(mutated_prot_emb_diff, edge_mat[:mutated_seq_length, :mutated_seq_length], rasp_models, variant_res_idx, device=device)
                        if Stability_pred is None:
                            print('Stability_pred failed',flush=True)
                            continue
                        all_stability_preds.append(Stability_pred)
                        
                        # for duplicate variant outputs (due to multiple interaction pairs)
                        stability_key = f"{refseq_id} {complex_and_vt.split(' ')[1]}"
                        stability_keys.append(stability_key)
                
                all_vt_ids.append(vt_id)
                all_wt_ids.append(complex_id)
                global_labels.append(global_label)
                ppi_preds.append(float(pred))
                
                print(f'done',flush=True)
                prediction_count += 1

                if prediction_count % 2000 == 1000:
                    np.save(f'{results_dir}/all_wt_ids.npy',all_wt_ids)
                    np.save(f'{results_dir}/all_vt_ids.npy',all_vt_ids)
                    np.save(f'{results_dir}/ppi_preds.npy',ppi_preds)
                    np.save(f'{results_dir}/all_stability_preds.npy',all_stability_preds)
                    np.save(f'{results_dir}/stability_keys.npy',stability_keys)
                    np.save(f'{results_dir}/global_labels.npy',global_labels)
                

            
            
            except Exception as e:
                print()
                print(f'error {e}',flush=True)
                print(e,flush=True)
                # sys.exit(1)
    
    '''
    SAVE NECESSARY INFORMATION FOR RE-CALIBRATION WITHOUT RE-RUNNING MODEL
    '''
    np.save(f'{results_dir}/all_wt_ids.npy',all_wt_ids)
    np.save(f'{results_dir}/all_vt_ids.npy',all_vt_ids)
    np.save(f'{results_dir}/ppi_preds.npy',ppi_preds)
    np.save(f'{results_dir}/all_stability_preds.npy',all_stability_preds)
    np.save(f'{results_dir}/stability_keys.npy',stability_keys)
    np.save(f'{results_dir}/global_labels.npy',global_labels)

    return ppi_preds, all_stability_preds, stability_keys, global_labels






    


