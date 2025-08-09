import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_add_pool
from torch_geometric.utils import dense_to_sparse
import glob
import numpy as np

'''
GAT_pool
'''
class GAT_pool(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, num_heads=4):
        super(GAT_pool, self).__init__()
        hidden_dim = 128
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat_out = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, output_dim)
        
    def forward(self, x, edge_index, batch):
        x = torch.relu(self.gat1(x, edge_index))
        x = self.gat_out(x, edge_index)
        x = global_mean_pool(x, batch)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        # no sigmoid with BCEWithLogitsLoss
        return x #torch.sigmoid(x)

'''
GAT_residue
'''
class GAT_residue(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super(GAT_residue, self).__init__()
        self.gat1 = GATConv(input_dim, 128, heads=num_heads, concat=True)
        self.gat_out = GATConv(128 * num_heads, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = torch.relu(self.gat1(x, edge_index))
        x = self.gat_out(x, edge_index)
        return x


'''
RaSP3-trained Stability predictor
'''
class GAT_RaSP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super(GAT_RaSP, self).__init__()
        self.gat1 = GATConv(input_dim, 128, heads=num_heads, concat=True)
        self.gat_out = GATConv(128 * num_heads, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = torch.relu(self.gat1(x, edge_index))
        x = self.gat_out(x, edge_index)
        return x

'''
stability fine-tuned ppi predictor
'''
# class GAT_mut_processor(nn.Module):
#     def __init__(self, input_dim, hidden_dim=64, output_dim=1,
#                  num_heads=4, mutation_diff_dim=1024):
#         super(GAT_mut_processor, self).__init__()


#         # Process mutation site difference
#         self.mutation_diff_processor = nn.Sequential(
#             nn.Linear(mutation_diff_dim, 128),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(128, 32)
#         )

#         # Single unified GAT for the complex
#         self.complex_gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
#         self.complex_gat2 = GATConv(hidden_dim * num_heads, hidden_dim // 2, heads=1, concat=False)

#         # Binding prediction with mutation diff features
#         self.binding_predictor = nn.Sequential(
#             nn.Linear(hidden_dim // 2 + 32, 16),  # +32 for processed mutation diff
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(16, output_dim)
#         )

#     def forward(self, x, edge_index, mutation_idx, num_mut_res, mutation_site_diff):

#         # Process mutation site difference
#         if mutation_site_diff.dim() == 1:
#             mutation_site_diff = mutation_site_diff.unsqueeze(0)
#         processed_mut_diff = self.mutation_diff_processor(mutation_site_diff)

#         # Process full complex
#         h = torch.relu(self.complex_gat1(x, edge_index))
#         h = torch.relu(self.complex_gat2(h, edge_index))


#         # Get features at mutation site
#         features_at_mutation = h[mutation_idx:mutation_idx+1]

#         # Combine GAT features with mutation difference
#         combined = torch.cat([features_at_mutation, processed_mut_diff], dim=-1)

#         return self.binding_predictor(combined)

class GAT_mut_processor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=1,
                 num_heads=4, mutation_diff_dim=1024):
        super(GAT_mut_processor, self).__init__()

        # Process mutation site difference
        self.mutation_diff_processor = nn.Sequential(
            nn.Linear(mutation_diff_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)
        )

        # Single unified GAT for the complex
        self.complex_gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.complex_gat2 = GATConv(hidden_dim * num_heads, hidden_dim // 2, heads=1, concat=False)

        # Binding prediction with mutation diff features
        self.binding_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 128, 128),  # +32 for processed mutation diff
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_dim)
        )

    def forward(self, x, edge_index, mutation_idx, mutation_site_diff):

        # Process mutation site difference
        if mutation_site_diff.dim() == 1:
            mutation_site_diff = mutation_site_diff.unsqueeze(0)
        processed_mut_diff = self.mutation_diff_processor(mutation_site_diff)

        # Process full complex
        h = torch.relu(self.complex_gat1(x, edge_index))
        h = torch.relu(self.complex_gat2(h, edge_index))


        # Get features at mutation site
        features_at_mutation = h[mutation_idx:mutation_idx+1]

        # Combine GAT features with mutation difference
        combined = torch.cat([features_at_mutation, processed_mut_diff], dim=-1)

        return self.binding_predictor(combined)

'''
function to initialize ppi models
'''
def get_ppi_models(model_dir, device, seq_confirmed_code=''):
    # new combined set, using mutated-prot-based clustering
    method = f'interaction_loss_combined_sahni_fragoza_varchamp1p_cava{seq_confirmed_code}_mclust'
    input_dim = 1025
    BATCH_SIZE = 16
    
    residue_models = []
    pool_models = []

    for model_mode in (0,1):
        RESIDUE_MODEL = model_mode
    
        complex_tag = '_simple'
            
        if RESIDUE_MODEL:
            model_paths = glob.glob(f'{model_dir}/gnn_prott5_{method}_batch_{BATCH_SIZE}{complex_tag}_residue_*.pt')
        else:
            model_paths = glob.glob(f'{model_dir}/gnn_prott5_{method}_batch_{BATCH_SIZE}{complex_tag}_pool_*.pt')
        
        for model_path in model_paths:
            fold = model_path.split('_')[-1].replace('.pt','')
        
            if RESIDUE_MODEL:
                model = GAT_residue(input_dim=input_dim, hidden_dim=1024, output_dim=1).to(device)
            else:
                model = GAT_pool(input_dim=input_dim, hidden_dim=1024, output_dim=1).to(device)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.eval()
            
            if RESIDUE_MODEL:
                residue_models.append(model)
            else:
                pool_models.append(model)

    return residue_models, pool_models

'''
get scaled stability finetune models
'''
def get_scaledstability_finetune_models(model_dir, device):
    # new combined set, using mutated-prot-based clustering
    input_dim = 1024
    BATCH_SIZE = 16
    
    models = []
    model_paths = glob.glob(f'{model_dir}/prod_models/PPI_combined_sahni_fragoza_varchamp1p_cava_simple_gmsd_scaledstability_finetune_fold_*.pt')
    
    for model_path in model_paths:
        model = GAT_mut_processor(input_dim=input_dim).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        models.append(model)
        
    assert len(models) != 0

    return models

'''
helper function to format input for ppi model
'''
def format_model_input(embedding, edge_mat, device):
    features = torch.tensor(embedding, dtype=torch.float).to(device)
    edge_index = torch.tensor(edge_mat)
    edge_index, _ = dense_to_sparse(edge_index)
    edge_index = edge_index.to(device)
    
    return features, edge_index
    
'''
ppi prediction function, averages prediction from 10 cross-validation models
'''
# def ppi_bagged_predict(embedding, edge_mat, models, RESIDUE_MODEL, variant_res_idx, device):
#     features, edge_index = format_model_input(embedding, edge_mat, device)
#     preds = []
#     # posts = []
#     for i,model in enumerate(models):
#         with torch.no_grad():
#             batch = torch.zeros(features.size(0), dtype=torch.long).to(device)
#             if RESIDUE_MODEL:
#                 pred = torch.sigmoid(model(features, edge_index)).squeeze(0).squeeze(1).cpu().numpy()
#             else:
#                 pred = torch.sigmoid(model(features, edge_index, batch)).squeeze(0).squeeze(0).cpu().numpy()
        
#         preds.append(pred)
        
#     mean_pred = np.mean(np.array(preds),axis=0)
    
#     # zero-based
#     return mean_pred[variant_res_idx] if RESIDUE_MODEL else mean_pred

def ppi_bagged_predict(embedding, edge_mat, models, RESIDUE_MODEL, variant_res_idx, device):
    try:
        features, edge_index = format_model_input(embedding, edge_mat, device)

        # Check for NaNs or empty tensors
        if features is None or edge_index is None:
            return None
        if features.size(0) == 0 or edge_index.size(1) == 0:
            return None

        preds = []

        for i, model in enumerate(models):
            with torch.no_grad():
                batch = torch.zeros(features.size(0), dtype=torch.long).to(device)

                # Validate indexing bounds for edge_index
                if edge_index.max() >= features.size(0):
                    print(f"[ERROR] edge_index out of bounds: max={edge_index.max()}, features={features.size(0)}")
                    return None

                if RESIDUE_MODEL:
                    out = model(features, edge_index)
                    if out.size(0) == 0 or out.size(1) == 0:
                        return None
                    pred = torch.sigmoid(out).squeeze(0).squeeze(1).cpu().numpy()
                else:
                    out = model(features, edge_index, batch)
                    if out.size(0) == 0:
                        return None
                    pred = torch.sigmoid(out).squeeze(0).squeeze(0).cpu().numpy()

                preds.append(pred)

        mean_pred = np.mean(np.array(preds), axis=0)

        if RESIDUE_MODEL:
            if not (0 <= variant_res_idx < len(mean_pred)):
                print(f"[ERROR] variant_res_idx {variant_res_idx} out of bounds for output of size {len(mean_pred)}")
                return None
            return mean_pred[variant_res_idx]
        else:
            return mean_pred

    except RuntimeError as e:
        if "indexSelectLargeIndex" in str(e):
            print(f"[CUDA INDEX ERROR] {e}")
            return None
        raise  # re-raise if it's not the known issue

'''
function to initialize stability models
'''
def get_stability_models(device):
    BAGGED = 0
    rasp_models = []
    rasp_model_paths = glob.glob(f"/data/ross/gnn/jose_2016_lossgain_models{'/bagged' if BAGGED else ''}/gnn_prott5_rasp3_Stability_residue_*.pt")
    for model_path in rasp_model_paths:
        fold = model_path.split('_')[-1].replace('.pt','')
        
        model = GAT_RaSP(input_dim=1024, hidden_dim=1024, output_dim=1).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        rasp_models.append(model)

    return rasp_models

'''
helper function to format input for stability model
'''
def RASP_format_model_input(embedding, edge_mat, device):
    features = torch.tensor(embedding, dtype=torch.float).to(device)
    edge_index = torch.tensor(edge_mat)
    edge_index, _ = dense_to_sparse(edge_index)
    edge_index = edge_index.to(device)
    
    return features, edge_index

'''
stability prediction function, averages prediction from 10 cross-validation models
'''
# def RASP_bagged_predict(embedding, edge_mat, models, variant_res_idx, device):
#     features, edge_index = RASP_format_model_input(embedding, edge_mat, device)
#     preds = []
#     for i,model in enumerate(models):
#         with torch.no_grad():
#             batch = torch.zeros(features.size(0), dtype=torch.long).to(device)
#             pred = torch.sigmoid(model(features, edge_index)).squeeze(0).squeeze(1).cpu().numpy()
            
#         preds.append(pred)
    
#     return float(np.mean(np.array(preds),axis=0)[variant_res_idx])

def RASP_bagged_predict(embedding, edge_mat, models, variant_res_idx, device):
    try:
        features, edge_index = RASP_format_model_input(embedding, edge_mat, device)

        # Validate input tensors
        if features is None or edge_index is None:
            return None
        if features.size(0) == 0 or edge_index.size(1) == 0:
            return None
        if edge_index.max() >= features.size(0):
            print(f"[ERROR] edge_index out of bounds: max={edge_index.max()}, features={features.size(0)}")
            return None

        preds = []

        for i, model in enumerate(models):
            with torch.no_grad():
                batch = torch.zeros(features.size(0), dtype=torch.long).to(device)

                out = model(features, edge_index)
                if out.size(0) == 0 or out.size(1) == 0:
                    return None

                pred = torch.sigmoid(out).squeeze(0).squeeze(1).cpu().numpy()

            preds.append(pred)

        mean_pred = np.mean(np.array(preds), axis=0)

        if not (0 <= variant_res_idx < len(mean_pred)):
            print(f"[ERROR] variant_res_idx {variant_res_idx} out of bounds for output of size {len(mean_pred)}")
            return None

        return float(mean_pred[variant_res_idx])

    except RuntimeError as e:
        if "indexSelectLargeIndex" in str(e):
            print(f"[CUDA INDEX ERROR] {e}")
            return None
        raise  # Re-raise if it's a different error


def ppi_stability_finetune_predict(embedding, edge_mat, models, mutation_idx, mutation_site_diff, device):
    try:
        features, edge_index = format_model_input(embedding, edge_mat, device)
        mutation_site_diff = torch.tensor(mutation_site_diff, dtype=torch.float).to(device)

        # Check for NaNs or empty tensors
        if features is None or edge_index is None:
            return None
        if features.size(0) == 0 or edge_index.size(1) == 0:
            return None

        preds = []

        for i, model in enumerate(models):
            with torch.no_grad():
                
                # Validate indexing bounds for edge_index
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







