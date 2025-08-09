'''
run_model_inference_pipeline_scaledstability_finetune.py
Author: Ross Stewart
Date: 07/19/2025

runs new fine-tuned stability model inference on a given dataset (e.g. clinvar, cosmic) and saves information needed for results analysis
'''

import sys
import numpy as np
import os
import pickle
from ppi_utils import run_inference_on_dataset, convert_one_to_zero_based_variant_dict
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


'''
PUT FASTA AND GRAPH DIR IN /data/ross/ppi_lossgain/interaction_loss/{dataset_name}/af3_graphs
and
/data/ross/ppi_lossgain/interaction_loss/{dataset_name}/{dataset_name}_{method}_wt_and_vt.fasta
'''
assert len(sys.argv) == 4, f'Usage: python {sys.argv[0]} <GPU> <DATASET_NAME> <PERCENT_KEEP>'
gpu = sys.argv[1]
dataset_name = sys.argv[2]
PERCENT_KEEP = float(sys.argv[3])
# SEQ_CONFIRMED_MODEL = int(sys.argv[4])
seq_confirmed_code = '_scaledstability_finetune'
# PRELOADED = int(sys.argv[3])
method = 'interaction_loss'

if not os.path.exists(f'/data/ross/ppi_lossgain/interaction_loss/{dataset_name}/af3_graphs') or not os.path.exists(f'/data/ross/ppi_lossgain/interaction_loss/{dataset_name}/{dataset_name}_{method}_wt_and_vt.fasta'):
    print(f'''
    PUT GRAPH DIR AND FASTA IN /data/ross/ppi_lossgain/interaction_loss/{dataset_name}/af3_graphs
    and
    /data/ross/ppi_lossgain/interaction_loss/{dataset_name}/{dataset_name}_{method}_wt_and_vt.fasta
    ''')
    sys.exit(1)

results_dir = f'./combined_sahni_fragoza_varchamp1p_cava{seq_confirmed_code}_results/{dataset_name}'
os.makedirs(results_dir,exist_ok=True)

data_dir = '/data/ross/ppi_lossgain/interaction_loss'
data_wd = f'{data_dir}/{dataset_name}'
ppi_wd = '/home/rcstewart/ppi_lossgain'
graph_dir = f'{data_wd}/af3_graphs'
t5_fasta_path = f'{data_wd}/{dataset_name}_{method}_wt_and_vt.fasta'


print('running_pipeline')
ppi_preds, all_stability_preds, stability_keys, global_labels = run_inference_on_dataset(gpu, dataset_name, graph_dir, t5_fasta_path, results_dir, percent_keep=PERCENT_KEEP, seq_confirmed_code=seq_confirmed_code) # main inference pipeline in ppi_utis.py





