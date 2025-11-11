#!/usr/bin/env python3
"""
AlphaFold3 Contact Graph Generator
Generates residue contact graphs from AlphaFold3 mmCIF structures.

Author: Ross Stewart, September 2025

Usage:
    python 01_make_contact_graphs_and_fasta.py <working_dir> <mmcif_dir> <variants_file> <n_jobs>
"""

import sys
import os
import glob
import pickle
import argparse
import numpy as np
import scipy.sparse as sp
from scipy.io import savemat, loadmat
from Bio import PDB
from Bio.PDB import MMCIFParser
from joblib import Parallel, delayed


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
EDGE_DIST_THRESHOLD = 4.5  # angstroms for any atom pair
THREE_LETTER_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D",
    "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
    "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
    "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Sec": "U", "Pyl": "O", "Asx": "B", "Glx": "Z",
    "Xaa": "X", "Ter": "*"
}


def get_labeled_residues(variant_file):
    """Parse variant file to get mutation positions and complex IDs."""
    all_pos_indices = {}
    all_complex_ids = set()
    n_variant_partner_interactions = 0
    
    with open(variant_file, 'r') as f:
        for line in f:
            id_a, variant, id_b = line.strip().split('\t')
            n_variant_partner_interactions += 1
            
            complex_id = f'{id_a}:{id_b}'
            
            # parse variant notation (e.g., V123A)
            wt_res = variant[0]
            res_idx = int(variant[1:-1]) - 1  # convert to 0-based
            mt_res = variant[-1]
            
            if complex_id not in all_pos_indices:
                all_pos_indices[complex_id] = []
            
            all_pos_indices[complex_id].append(('A', res_idx, wt_res, mt_res))
            all_complex_ids.add(complex_id)
    
    return all_pos_indices, all_complex_ids, n_variant_partner_interactions


def find_mmcif_file(id_a, id_b, mmcif_dir):
    """
    Find mmCIF file containing both protein IDs in the filename.
    Returns (filepath, swapped) where swapped indicates if IDs are reversed.
    """
    found_files = []
    
    # search for all .cif and .mmcif files in directory
    for ext in ['.cif', '.mmcif']:
        pattern = os.path.join(mmcif_dir, f'*{ext}')
        for filepath in glob.glob(pattern):
            filename = os.path.basename(filepath).replace(ext, '')
            
            # check all possible ID combinations
            id_combinations = [
                (f"{id_a}_{id_b}", False),
                (f"{id_b}_{id_a}", True),
                (f"{id_a.lower()}_{id_b.lower()}", False),
                (f"{id_b.lower()}_{id_a.lower()}", True),
                (f"{id_a}_{id_b.lower()}", False),
                (f"{id_a.lower()}_{id_b}", False),
                (f"{id_b}_{id_a.lower()}", True),
                (f"{id_b.lower()}_{id_a}", True)
            ]
            
            for id_pattern, swapped in id_combinations:
                if id_pattern in filename:
                    # verify it's actually the IDs and not part of a longer string by checking for word boundaries (underscore or start/end)
                    idx = filename.find(id_pattern)
                    valid = True
                    
                    # check character before (if exists)
                    if idx > 0 and filename[idx-1] not in ['_', '-']:
                        valid = False
                    
                    # check character after (if exists)  
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
        raise FileNotFoundError(f"No mmCIF file found for {id_a} and {id_b}")
    elif len(unique_files) > 1:
        raise ValueError(f"Multiple mmCIF files found for {id_a} and {id_b}: {list(unique_files.keys())}")
    
    return list(unique_files.values())[0]


def make_graph(complex_id, mmcif_dir, save_dir):
    """Generate contact graph from mmCIF structure."""
    id_a, id_b = complex_id.split(':')
    out_file = os.path.join(save_dir, f'{id_a}_{id_b}.mat')
    
    if os.path.exists(out_file):
        return
    
    # find and parse structure file
    mmcif_file, swapped = find_mmcif_file(id_a, id_b, mmcif_dir)
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(complex_id, mmcif_file)
    
    # assign chains based on swap status
    if not swapped:
        model_a = structure[0]['A']
        model_b = structure[0]['B']
    else:
        model_a = structure[0]['B']  # id_a gets chain B when swapped
        model_b = structure[0]['A']  # id_b gets chain A when swapped
    
    # extract residue coordinates
    aa_labels = []
    coords = []
    num_residues_a = 0
    
    # process chain A (id_a)
    for residue in model_a:
        if PDB.is_aa(residue):
            num_residues_a += 1
            aa_labels.append(THREE_LETTER_TO_ONE[residue.get_resname().capitalize()])
            atom_coords = {atom.get_name(): atom.coord for atom in residue}
            coords.append(atom_coords)
    
    # process chain B (id_b)
    for residue in model_b:
        if PDB.is_aa(residue):
            aa_labels.append(THREE_LETTER_TO_ONE[residue.get_resname().capitalize()])
            atom_coords = {atom.get_name(): atom.coord for atom in residue}
            coords.append(atom_coords)
    
    num_residues = len(aa_labels)
    
    # build contact matrix
    edge_mat = np.zeros((num_residues, num_residues))
    
    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            if coords[i] and coords[j]:
                # check for close atom pairs
                for atom_i in coords[i]:
                    for atom_j in coords[j]:
                        dist = np.linalg.norm(coords[i][atom_i] - coords[j][atom_j])
                        if dist <= EDGE_DIST_THRESHOLD:
                            edge_mat[i, j] = edge_mat[j, i] = 1
                            break
                    if edge_mat[i, j]:
                        break
    
    # save graph data
    savemat(out_file, {
        'G': sp.csr_matrix(edge_mat),
        'L': aa_labels,
        'NRR': num_residues_a
    })


def write_variant_labels(variant_indices, save_dir, method='interaction_loss'):
    """Generate variant sequence files."""
    variant_labels_lines = []
    variant_labels_sep_lines = []
    num_bad_variants = 0
    
    for complex_id in variant_indices:
        id_a, id_b = complex_id.split(':')
        labels_file = os.path.join(save_dir, f'{id_a}_{id_b}.labels_separated')
        
        if not os.path.exists(labels_file):
            continue
        
        # read wild-type sequence
        with open(labels_file, 'r') as f:
            pdb_seq = f.read().strip()
        
        # read number of residues in chain A
        with open(os.path.join(save_dir, f'{id_a}_{id_b}.num_residues_a'), 'r') as f:
            num_residues_a = int(f.read().strip())
        
        chain_to_pos = {}
        
        for chain, mt_idx, wt_res, mt_res in variant_indices[complex_id]:
            # validate mutation
            if mt_idx >= num_residues_a or pdb_seq[mt_idx] != wt_res:
                print(chain, mt_idx, wt_res, mt_res, num_residues_a, '\n', pdb_seq[mt_idx], wt_res, '\n', pdb_seq)
                num_bad_variants += 1
                continue
            
            # create variant sequence
            vt_seq = list(pdb_seq)
            vt_seq[mt_idx] = mt_res
            vt_seq = ''.join(vt_seq)
            
            # store variant sequences
            variant_name = f'{id_a}_{id_b}_{method}_variant_{wt_res}{mt_idx}{mt_res}'
            variant_labels_lines.append(f'>{variant_name}\n{vt_seq.upper()}\n')
            variant_labels_sep_lines.append(f'>{variant_name}\n{vt_seq}\n')
            
            if chain not in chain_to_pos:
                chain_to_pos[chain] = ''
            chain_to_pos[chain] += f'{mt_idx}\t{wt_res}{mt_idx}{mt_res}\n'
        
        # write position files
        for chain in chain_to_pos:
            assert chain == 'A'  # only chain A mutations
            pos_file = os.path.join(save_dir, f'{id_a}_{id_b}.{method}_pos')
            with open(pos_file, 'w') as f:
                f.write(chain_to_pos[chain])
    
    print(f'{num_bad_variants} bad variants')
    
    # write all variant sequences
    variant_labels_file = os.path.join(save_dir, 'all_variants.labels')
    variant_labels_sep_file = os.path.join(save_dir, 'all_variants.labels_separated')
    
    with open(variant_labels_file, 'w') as f:
        f.writelines(variant_labels_lines)
    
    with open(variant_labels_sep_file, 'w') as f:
        f.writelines(variant_labels_sep_lines)
    
    return variant_labels_sep_file


def generate_fasta_output(save_dir, wd, variant_labels_sep_file):
    """Generate FASTA file with wild-type and variant sequences."""
    # parse variant sequences
    wt_to_vt_seq = {}
    with open(variant_labels_sep_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                filename_key = line[1:].strip()
                pdb_id = filename_key.split('_')[0]
                if pdb_id not in wt_to_vt_seq:
                    wt_to_vt_seq[pdb_id] = {}
                variant = filename_key.split('_')[-1]
            else:
                seq = line.strip()
                wt_to_vt_seq[pdb_id][variant] = seq
    
    # write FASTA output
    seen_wts = set()
    seen_vts = set()
    seen_partners = set()
    
    with open(os.path.join(wd, 'wt_and_vt.fasta'), 'w') as f_out:
        seen_pdbs = set()
        
        # process all variant files
        variant_files = glob.glob(f"{save_dir}/*.interaction_loss_pos") + \
                       glob.glob(f"{save_dir}/*.interaction_loss_neg")
        
        for f_pos in variant_files:
            labels_file = f_pos.replace('interaction_loss_pos', 'labels_separated')\
                              .replace('interaction_loss_neg', 'labels_separated')
            
            if not os.path.exists(labels_file.replace('labels_separated', 'mat')):
                continue
            
            pdb_id = os.path.basename(labels_file).split('.')[0]
            if pdb_id in seen_pdbs:
                continue
            seen_pdbs.add(pdb_id)
            
            wt_id = pdb_id.split('_')[0]
            partner_id = '_'.join(pdb_id.split('_')[1:])
            
            # parse sequences
            with open(labels_file, 'r') as f:
                line = f.read().strip()
                wt_seq = ''
                partner_seq = ''
                
                for char in line:
                    if char.islower():
                        partner_seq += char
                    else:
                        wt_seq += char
            
            # write wild-type sequence
            if wt_id not in seen_wts:
                f_out.write(f">{wt_id}\n{wt_seq}\n")
                seen_wts.add(wt_id)
            
            # write partner sequence
            if partner_id not in seen_partners:
                f_out.write(f">{partner_id}\n{partner_seq.upper()}\n")
                seen_partners.add(partner_id)
            
            # write variant sequences
            if wt_id in wt_to_vt_seq:
                for variant, vt_partner_seq in wt_to_vt_seq[wt_id].items():
                    vt_id = f'{wt_id} {variant}'
                    
                    if vt_id not in seen_vts:
                        vt_seq = ''
                        for char in vt_partner_seq:
                            if not char.islower():
                                vt_seq += char
                        
                        f_out.write(f">{vt_id}\n{vt_seq}\n")
                        seen_vts.add(vt_id)


if __name__ == "__main__":
    # load variant data
    variant_indices, complex_ids, n_variant_partner_interactions = get_labeled_residues(variants_file)
    print(f'Loaded {n_variant_partner_interactions} variant-partner interactions')
    
    # generate contact graphs in parallel
    print(f'Generating {len(complex_ids)} contact graph{"s" if len(complex_ids) != 1 else ""}...')
    Parallel(n_jobs=n_jobs)(
        delayed(make_graph)(complex_id, mmcif_dir, save_dir) 
        for complex_id in complex_ids
    )
    
    # process graph files and extract sequences
    for mat_file in glob.glob(f'{save_dir}/*.mat'):
        key = os.path.splitext(os.path.basename(mat_file))[0]
        
        # load graph data
        mat_data = loadmat(mat_file)
        dense_mat = mat_data['G'].toarray()
        pdb_seq = mat_data['L']
        num_residues_a = mat_data['NRR'].item()
        
        # process sequence
        if len(pdb_seq) == 1:
            pdb_seq = pdb_seq[0]
        else:
            pdb_seq = ''.join(pdb_seq)
        
        # write sequence files
        with open(os.path.join(save_dir, f'{key}.labels'), 'w') as f:
            f.write(pdb_seq)
            print(f"Wrote {save_dir}/{key}.labels")
        
        with open(os.path.join(save_dir, f'{key}.labels_separated'), 'w') as f:
            f.write(pdb_seq[:num_residues_a] + pdb_seq[num_residues_a:].lower())
            print(f"Wrote {save_dir}/{key}.labels_separated")
        
        with open(os.path.join(save_dir, f'{key}.num_residues_a'), 'w') as f:
            f.write(str(num_residues_a))
    
    # generate variant labels
    variant_labels_sep_file = write_variant_labels(variant_indices, save_dir)
    
    # generate final FASTA output
    generate_fasta_output(save_dir, wd, variant_labels_sep_file)
