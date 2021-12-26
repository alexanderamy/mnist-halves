import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

def compute_sims(model, dataset, metric, device):
    '''
    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    for available metrics
    '''
    model.eval()
    act_targets = []
    est_targets = []
    for _, (X, Y, _) in enumerate(dataset):
        X, Y = X.to(device), Y.to(device)
        _, _, _, _, V_yW_xX, _, _, _, _, Y = model(X, Y)
        act_targets.append(Y)
        est_targets.append(V_yW_xX)
    act_targets = torch.cat(act_targets).detach().numpy()
    est_targets = torch.cat(est_targets).detach().numpy()
    sims = distance.cdist(act_targets, est_targets, metric)
    return sims

def get_matches(sims):
    row_ind, col_ind = linear_sum_assignment(sims)
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
            num = (dataset.dataset.labels[:head] == dataset.dataset.labels[:head][col_ind]).sum().item()
        else:
            num = (dataset.labels == dataset.labels[col_ind]).sum().item()
        tot = row_ind.shape[0]
        acc = num / tot
        print(f'{acc:.1%}')
    elif eval_type == 'label':
        for i in range(10):
            if isinstance(dataset, torch.utils.data.dataset.Subset):
                head = len(dataset)
                num = ((dataset.dataset.labels[:head] - i == 0) * (dataset.dataset.labels[:head][col_ind] - i == 0)).sum().item()
                tot = np.unique(dataset.dataset.labels[:head], return_counts=True)[1][i]
            else:
                num = ((dataset.labels - i == 0) * (dataset.labels[col_ind] - i == 0)).sum().item() 
                tot = np.unique(dataset.labels, return_counts=True)[1][i]
            acc = num / tot
            print(f'{i}: {acc:.1%}')

