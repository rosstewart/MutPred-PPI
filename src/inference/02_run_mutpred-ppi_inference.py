'''
Step 3 of the MutPred-PPI inference pipeline: score every variant.

Reads the contact graphs and `wt_and_vt.fasta` written by step 2 and writes
`<working_dir>/results/MutPred-PPI_preds.tsv` with columns
`interactor, partner, mutation, score` (mutation is 1-based).

Interaction prediction labels:
- 1: Disrupted interaction (variant disrupts protein-protein interaction)
- 0: Unperturbed interaction (variant maintains wild-type interaction)

Usage:
    python 02_run_mutpred-ppi_inference.py <working_dir> [--device DEVICE] [--models-dir PATH]
'''

import argparse
import os
import sys
import warnings

from sklearn.exceptions import InconsistentVersionWarning

from inference.pipeline.inference_utils import run_inference_on_dataset

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

_DEFAULT_MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../', 'weights'))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Run MutPred-PPI inference')
    parser.add_argument('working_dir',
                        help='Working directory containing af3_graphs/ and wt_and_vt.fasta')
    parser.add_argument('--device', default='cuda:0',
                        help='Compute device (default: cuda:0; falls back to CPU '
                             'if CUDA is unavailable)')
    parser.add_argument('--models-dir', default=_DEFAULT_MODELS_DIR,
                        help='Directory holding MutPred-PPI.pt and '
                             'mutation_diff_scaler.pkl (default: weights/)')
    parser.add_argument('--arch', choices=('current', 'v1.0'), default='current',
                        help="Model generation. 'current' loads MutPred-PPI.pt. 'v1.0' "
                             "loads the published 10-checkpoint ensemble (RECOMB 2026 / "
                             "bioRxiv v1-v2) with its own architecture, from --models-dir "
                             "together with that generation's mutation_diff_scaler.pkl. "
                             "Scores from the two are not comparable.")
    return parser.parse_args(argv)


def main(argv=None):
    # Set before any CUDA context is created, so it must stay inside main()
    # rather than at import: this module used to run argparse and mutate the
    # environment at import time, with no __main__ guard at all, which meant
    # merely importing it parsed sys.argv and exited on anything unexpected.
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    args = parse_args(argv)
    working_dir = args.working_dir
    graph_dir = f'{working_dir}/af3_graphs'
    t5_fasta_path = f'{working_dir}/wt_and_vt.fasta'
    results_dir = f'{working_dir}/results'
    os.makedirs(results_dir, exist_ok=True)

    missing = [p for p in (graph_dir, t5_fasta_path) if not os.path.exists(p)]
    if missing:
        print(f"Error: missing input(s): {', '.join(missing)}\n"
              f"  Run step 2 (01_make_contact_graphs_and_fasta.py) first.",
              file=sys.stderr)
        return 1

    run_inference_on_dataset(args.device, working_dir, graph_dir, t5_fasta_path,
                             results_dir, models_dir=args.models_dir, arch=args.arch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
