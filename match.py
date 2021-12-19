import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

def make_sims_matrix(model, dataset, n=np.inf):
    model.eval()
    sims = []
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    for i in tqdm(range(min(n, len(dataset)))):
        X_i = dataset[i][0]
        sims_i = []
        est_target = X_i.reshape(1, 14 * 28)
        est_target = est_target @ model.W_x.weight.t()
        est_target = est_target @ model.V_y.weight.t()
        for _, (_, Y_j, label_j) in enumerate(dataset):
            pot_match = Y_j.reshape(1, 14 * 28)
            sims_i.append(max(0.0, cos(est_target, pot_match).item()))
        sims.append(sims_i)
    return sims

def get_matches(sims):
    row_ind, col_ind = linear_sum_assignment(sims, maximize=True)
    return row_ind, col_ind

def eval_matches(dataset, row_ind, col_ind, eval_type='exact'):
    if eval_type == 'exact':
        num = (row_ind == col_ind).sum().item()
        tot = row_ind.shape[0]
        acc = num / tot
        print(f'{acc:.1%}')
    elif eval_type == 'correct':
        if isinstance(dataset, torch.utils.data.dataset.Subset):
            head = len(dataset)
            num = (dataset.dataset.targets[:head] == dataset.dataset.targets[:head][col_ind]).sum().item()
        else:
            num = (dataset.targets == dataset.targets[col_ind]).sum().item()
        tot = row_ind.shape[0]
        acc = num / tot
        print(f'{acc:.1%}')
    elif eval_type == 'label':
        for i in range(10):
            if isinstance(dataset, torch.utils.data.dataset.Subset):
                head = len(dataset)
                num = ((dataset.dataset.targets[:head] - i == 0) * (dataset.dataset.targets[:head][col_ind] - i == 0)).sum().item()
                tot = np.unique(dataset.dataset.targets[:head], return_counts=True)[1][i]
            else:
                num = ((dataset.targets - i == 0) * (dataset.targets[col_ind] - i == 0)).sum().item() 
                tot = np.unique(dataset.targets, return_counts=True)[1][i]
            acc = num / tot
            print(f'{i}: {acc:.1%}')

