import torch
from torch_geometric.utils import dense_to_sparse
import glob
import numpy as np


# Single definition lives in src/model.py; see its docstring for why.
import sys as _sys
from pathlib import Path as _Path
from model import MutPred_PPI  # noqa: E402,F401


def get_models(model_dir, device):
    input_dim = 1024

    models = []
    primary = f'{model_dir}/MutPred-PPI.pt'
    if glob.glob(primary):
        model_paths = [primary]
    else:
        # Ensemble directory (e.g. per-fold checkpoints): load every matching file.
        model_paths = glob.glob(f'{model_dir}/MutPred-PPI_*_megascale_all_*.pt')

    for model_path in model_paths:
        model = MutPred_PPI(input_dim=input_dim).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        model.eval()
        models.append(model)

    assert len(models) != 0, f"No model checkpoints found in {model_dir}"
    return models


# helper function to format input for ppi model
def format_model_input(embedding, edge_mat, device):
    features = torch.tensor(embedding, dtype=torch.float).to(device)
    edge_index = torch.tensor(edge_mat)
    edge_index, _ = dense_to_sparse(edge_index)
    edge_index = edge_index.to(device)
    return features, edge_index


def model_predict_subgraph(node_emb, edge_index_np, models, mut_local_idx, mutation_site_diff_np, device):
    """Run ensemble inference on a pre-built 2-hop subgraph.

    Args:
        node_emb:              (k, 1024) float32 numpy — subgraph node features
        edge_index_np:         (2, e) int32 numpy — COO edges in local coords
        models:                list of MutPred_PPI models
        mut_local_idx:         int — mutation site index within local nodes
        mutation_site_diff_np: (1024,) float32 numpy — scaled mutation diff
        device:                torch.device
    """
    try:
        x = torch.tensor(node_emb, dtype=torch.float).to(device)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long).to(device)
        mut_diff_t = torch.tensor(mutation_site_diff_np, dtype=torch.float).to(device)

        if x.size(0) == 0 or edge_index.size(1) == 0:
            return None
        if edge_index.max() >= x.size(0):
            print(f"[ERROR] subgraph edge_index out of bounds: "
                  f"max={edge_index.max()}, nodes={x.size(0)}")
            return None

        preds = []
        for model in models:
            with torch.no_grad():
                out = model(x, edge_index, mut_local_idx, mut_diff_t)
                if out.size(0) == 0:
                    return None
                preds.append(torch.sigmoid(out).squeeze().cpu().numpy())

        return float(np.mean(preds))

    except RuntimeError as e:
        if "indexSelectLargeIndex" in str(e):
            print(f"[CUDA INDEX ERROR] {e}")
            return None
        raise


def model_predict(embedding, edge_mat, models, mutation_idx, mutation_site_diff, device):
    try:
        features, edge_index = format_model_input(embedding, edge_mat, device)
        mutation_site_diff = torch.tensor(mutation_site_diff, dtype=torch.float).to(device)

        if features is None or edge_index is None:
            return None
        if features.size(0) == 0 or edge_index.size(1) == 0:
            return None

        preds = []

        for i, model in enumerate(models):
            with torch.no_grad():
                if edge_index.max() >= features.size(0):
                    print(f"[ERROR] edge_index out of bounds: max={edge_index.max()}, features={features.size(0)}")
                    return None

                out = model(features, edge_index, mutation_idx, mutation_site_diff)
                if out.size(0) == 0:
                    return None
                pred = torch.sigmoid(out).squeeze().cpu().numpy()
                preds.append(pred)

        mean_pred = np.mean(np.array(preds), axis=0)
        return mean_pred

    except RuntimeError as e:
        if "indexSelectLargeIndex" in str(e):
            print(f"[CUDA INDEX ERROR] {e}")
            return None
        raise
