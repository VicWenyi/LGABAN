import torch
from rdkit import Chem
from rdkit.Chem import Draw
from torch.utils.data import DataLoader
from dataloader import DTIDataset
from models import LGABAN
import pandas as pd
import numpy as np
from utils import graph_collate_func
import heapq
from configs import get_cfg_defaults


def get_att_values_and_features(model, data_loader, device):
    model.eval()
    att_values = []
    vd_values = []
    vp_values = []

    with torch.no_grad():
        for batch in data_loader:
            v_d, v_p, _ = batch
            v_d = v_d.to(device)
            v_p = v_p.to(device)

            v_d_out, v_p_out, _, att = model(v_d, v_p, mode="eval")

            att = att.detach().cpu().numpy()
            v_d_out = v_d_out.detach().cpu().numpy()
            v_p_out = v_p_out.detach().cpu().numpy()

            att_values.extend(att)
            vd_values.extend(v_d_out)
            vp_values.extend(v_p_out)

    return att_values, vd_values, vp_values

def get_top_percentile_indices(att_values, percentile=20):
    """Get the indices of the top percentile attention values."""
    # Compute the number of top elements to select
    num_top_elements = int(np.ceil(percentile / 100 * att_values.size))

    # Get the indices that would sort the array
    sorted_indices = np.argsort(att_values)

    # Select the indices of the top elements
    top_indices = sorted_indices[-num_top_elements:]

    return top_indices

cfg = get_cfg_defaults()
model = LGABAN(**cfg)

state_dict = torch.load(r'./result/BindingDB1/best_model_epoch_89.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

model.eval()

test_data_path = r'interpretation/3/interpretation-sample3.csv'
test_df = pd.read_csv(test_data_path).dropna()
test_list_IDs = range(len(test_df))

test_data = DTIDataset(test_list_IDs, test_df, max_drug_nodes=290)
test_loader = DataLoader(test_data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False,
                         num_workers=cfg.SOLVER.NUM_WORKERS, collate_fn=graph_collate_func)

device = torch.device("cpu" if torch.cuda.is_available() else "cuda")
test_df = pd.read_csv(test_data_path)
test_list_IDs = test_df.index.tolist()

test_data = DTIDataset(test_list_IDs, test_df, max_drug_nodes=290)
test_loader = DataLoader(test_data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False,
                         num_workers=cfg.SOLVER.NUM_WORKERS, collate_fn=graph_collate_func)


att_values, vd_values, vp_values = get_att_values_and_features(model, test_loader, device)
np.save("interpretation/3/att_values.npy", att_values)
np.save("interpretation/3/vd_values.npy", vd_values)
np.save("interpretation/3/vp_values.npy", vp_values)

print("att_values shape:", np.array(att_values).shape)
print("vd_values shape:", np.array(vd_values).shape)
print("vp_values shape:", np.array(vp_values).shape)

# Convert att_values to a NumPy array
att_values = np.array(att_values)
att_mean = np.mean(att_values, axis=1)
# 获取att中最大的10个权重值及其对应的位置
att_flattened = att_mean.reshape(-1)
top_10_indices = heapq.nlargest(10, range(len(att_flattened)), att_flattened.take)
top_10_values = att_flattened[top_10_indices]

# 将位置转换为att矩阵中的二维坐标
top_10_coords = [divmod(index, 290) for index in top_10_indices]

# 获取对应的vd和vp值
top_10_vd = [vd_values[0][coord[1]] for coord in top_10_coords]
top_10_vp = [vp_values[0][coord[0]] for coord in top_10_coords]

# 打印结果
for i in range(10):
     print(f"Top {i+1}:")
     print(f"Value: {top_10_values[i]}")
     print(f"Position in att matrix: {top_10_coords[i]}")
     print(f"vd: {top_10_vd[i]}")
     print(f"vp: {top_10_vp[i]}\n")

att_values = np.load("interpretation/3/att_values.npy")

for index in range(len(test_df)):
    # Select the appropriate row of the att_values matrix
    current_att_values = att_values[index, :, :]

    # Load the SMILES code from the interpretation-sample1.csv file
    smiles_code = test_df.iloc[index, 0]

    # Create a molecule object from the SMILES code
    mol = Chem.MolFromSmiles(smiles_code)

    # Get the number of actual atoms in the molecule
    num_atoms = mol.GetNumAtoms()

    # Get the attention weights for the actual atoms only
    actual_att_weights = current_att_values.flatten()[:num_atoms]

    # Find the indices for the top 20% important attention values
    important_indices = get_top_percentile_indices(actual_att_weights, percentile=20)

    # Define the highlight colors for atoms
    highlight_colors = {}
    for i in important_indices:
        # Convert numpy.int64 to int
        i = int(i)
        highlight_colors[i] = (0.5, 1, 0.5, 1)

    # Display the molecule with highlighted atoms
    img = Draw.MolToImage(mol, highlightAtoms=list(highlight_colors.keys()), highlightAtomColors=highlight_colors, size=(1024, 1024))
    img.show()
    img.save(f"./interpretation/highlighted_molecule_{index}.png")

