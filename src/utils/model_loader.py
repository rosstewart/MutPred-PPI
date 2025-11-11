import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_add_pool
from torch_geometric.utils import dense_to_sparse
import glob
import numpy as np


class GATMutPPI(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=1,
                 num_heads=4, mutation_diff_dim=1024):
        super(GATMutPPI, self).__init__()

        # process mutation site difference
        self.mutation_diff_processor = nn.Sequential(
            nn.Linear(mutation_diff_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)
        )

        # single unified GAT for the complex
        self.complex_gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.complex_gat2 = GATConv(hidden_dim * num_heads, hidden_dim // 2, heads=1, concat=False)

        # binding prediction with mutation diff features
        self.binding_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 128, 128),  # +128 for processed mutation diff
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_dim)
        )

    def forward(self, x, edge_index, mutation_idx, mutation_site_diff):

        # process mutation site difference
        if mutation_site_diff.dim() == 1:
            mutation_site_diff = mutation_site_diff.unsqueeze(0)
        processed_mut_diff = self.mutation_diff_processor(mutation_site_diff)

        # process full complex
        h = torch.relu(self.complex_gat1(x, edge_index))
        h = torch.relu(self.complex_gat2(h, edge_index))


        # get features at mutation site
        features_at_mutation = h[mutation_idx:mutation_idx+1]

        # combine GAT features with mutation difference
        combined = torch.cat([features_at_mutation, processed_mut_diff], dim=-1)

        return self.binding_predictor(combined)

def get_models(model_dir, device):
    # new combined set, using mutated-prot-based clustering
    input_dim = 1024
    BATCH_SIZE = 16
    
    models = []
    model_paths = glob.glob(f'{model_dir}/GATMutPPI_sahni_fragoza_varchamp_cava_*.pt')
    
    for model_path in model_paths:
        model = GATMutPPI(input_dim=input_dim).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        model.eval()

        models.append(model)
        
    assert len(models) != 0

    return models

# helper function to format input for ppi model
def format_model_input(embedding, edge_mat, device):
    features = torch.tensor(embedding, dtype=torch.float).to(device)
    edge_index = torch.tensor(edge_mat)
    edge_index, _ = dense_to_sparse(edge_index)
    edge_index = edge_index.to(device)
    
    return features, edge_index


def model_predict(embedding, edge_mat, models, mutation_idx, mutation_site_diff, device):
    try:
        features, edge_index = format_model_input(embedding, edge_mat, device)
        mutation_site_diff = torch.tensor(mutation_site_diff, dtype=torch.float).to(device)

        # check for NaNs or empty tensors
        if features is None or edge_index is None:
            return None
        if features.size(0) == 0 or edge_index.size(1) == 0:
            return None

        preds = []

        for i, model in enumerate(models):
            with torch.no_grad():
                
                # validate indexing bounds for edge_index
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
        raise  # re-raise if it's not the known issue







