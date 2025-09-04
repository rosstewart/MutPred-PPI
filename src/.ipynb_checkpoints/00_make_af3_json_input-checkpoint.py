#!/usr/bin/env python3
"""
AlphaFold3 Input Preprocessing
Generate JSON input files for AlphaFold3 from FASTA files and variant triplet TSV.
Author: Ross Stewart, September 2025

Usage:
    python 00_make_af3_input_file.py <fasta_file> <triplet_tsv> <output_directory>
"""

import argparse
import json
import os
import sys
import csv
from Bio import SeqIO


def parse_fasta(fasta_file):
    """Parse FASTA file and return dictionary of all sequences."""
    sequences = {}
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq_id = record.id
        seq = str(record.seq).upper()
        
        # Check that IDs don't contain hyphens (reserved for complex naming)
        if '-' in seq_id:
            raise ValueError(f"Sequence ID '{seq_id}' cannot contain hyphens (-)")
        
        # Sequence validation
        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
        invalid_chars = set(seq) - valid_aas
        if invalid_chars:
            raise ValueError(f"Sequence {seq_id} contains invalid characters: {invalid_chars}")
        
        sequences[seq_id] = seq
    
    return sequences


def parse_triplet_tsv(tsv_file):
    """Parse TSV file containing triplets (id_a, variant, id_b)."""
    complex_ids = set()
    
    with open(tsv_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        
        for row in reader:
            if len(row) >= 3:
                id_a, variant, id_b = row[0], row[1], row[2]
                # normalize pair order to ensure (A,B) and (B,A) are treated as same
                pair = tuple(sorted([id_a, id_b]))
                complex_ids.add(pair)
    
    return complex_ids


def create_af3_json(id_a, seq_a, id_b, seq_b):
    """Create AlphaFold3 JSON input structure."""
    complex_id = f"{id_a}-{id_b}"
    
    data = {
        "name": complex_id,
        "modelSeeds": [1],  # Single seed (5 samples) for speed
        "sequences": [
            {
                "protein": {
                    "sequence": seq_a,
                    "id": id_a
                }
            },
            {
                "protein": {
                    "sequence": seq_b,
                    "id": id_b
                }
            }
        ]
    }
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Generate AlphaFold3 input JSONs from FASTA file and variant triplet TSV'
    )
    parser.add_argument('fasta_file', help='Input FASTA file with protein sequences')
    parser.add_argument('triplet_tsv', help='TSV file with triplets (id_a, variant, id_b)')
    parser.add_argument('output_dir', help='Output directory for JSON files')
    parser.add_argument('--seeds', type=int, default=1, 
                       help='Number of AlphaFold3 model seeds (1-5, default: 1)')
    
    args = parser.parse_args()
    
    # validate seeds
    if not 1 <= args.seeds <= 5:
        print("Error: seeds must be between 1 and 5")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # parse FASTA to get all sequences
        print(f"Parsing FASTA file: {args.fasta_file}")
        sequences = parse_fasta(args.fasta_file)
        print(f"Loaded {len(sequences)} sequences")
        
        # parse triplet TSV to get unique pairs
        print(f"Parsing triplet TSV: {args.triplet_tsv}")
        complex_ids = parse_triplet_tsv(args.triplet_tsv)
        print(f"Found {len(complex_ids)} unique protein pairs")
        
        # process each unique pair
        successful = 0
        failed = []
        for id_a, id_b in complex_ids:
            try:
                # check if both IDs exist in FASTA
                if id_a not in sequences:
                    raise ValueError(f"ID '{id_a}' not found in FASTA file")
                if id_b not in sequences:
                    raise ValueError(f"ID '{id_b}' not found in FASTA file")
                
                seq_a = sequences[id_a]
                seq_b = sequences[id_b]
                
                json_data = create_af3_json(id_a, seq_a, id_b, seq_b)
                
                # update seeds if specified
                if args.seeds > 1:
                    json_data["modelSeeds"] = list(range(1, args.seeds + 1))
                
                # save JSON file
                complex_id = json_data["name"]
                output_path = os.path.join(args.output_dir, f"{complex_id}.json")
                
                with open(output_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                
                successful += 1
                print(f"  Created: {complex_id}.json ({len(seq_a)} + {len(seq_b)} aa)")
                
            except Exception as e:
                failed.append((id_a, id_b, str(e)))
                print(f"  Failed: {id_a}-{id_b} - {e}")
        
           
        print(f"Successfully created {successful} JSON files")
        if failed:
            print(f"Failed to process {len(failed)} pairs:")
            for id_a, id_b, error in failed:
                print(f"  {id_a}-{id_b}: {error}")
        print(f"Output directory: {args.output_dir}")
        print(f"Seeds per model: {args.seeds}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
