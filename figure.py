import numpy as np
from matplotlib import pyplot as plt

def make_figure(model, data, dataset, row_ind, col_ind, sample='random', n=5):
    if sample == 'random':
        idxs = np.random.choice(row_ind, n, replace=False)
    elif sample == 'correct':
        correct = (dataset.targets == dataset.targets[col_ind]).numpy()
        correct = np.where(correct == True)[0]
        idxs = np.random.choice(correct, n, replace=False)
    elif sample == 'cherrypicked':
        cherrypicked = row_ind == col_ind
        cherrypicked = np.where(cherrypicked == True)[0]
        idxs = np.random.choice(cherrypicked, n, replace=False)
    plots = ['top', 'ground truth bottom', 'estimated bottom', 'retrieved bottom']
    fig, axes = plt.subplots(4, len(idxs), figsize=(28, 14), gridspec_kw = {'wspace':0, 'hspace':0})
    for i in range(len(plots)):
        for j in range(len(idxs)):
            ax = axes[i, j]
            img_idx = idxs[j]
            if plots[i] == 'top':
                top_gt = data[img_idx][0]
                top_gt = top_gt.reshape(14, 28)
                ax.imshow(top_gt, cmap='gray', interpolation='none')
            elif plots[i] == 'ground truth bottom':
                bottom_gt = data[img_idx][1]
                bottom_gt = bottom_gt.reshape(14, 28)
                ax.imshow(bottom_gt, cmap='gray', interpolation='none')
            elif plots[i] == 'estimated bottom':
                top_gt = data[img_idx][0]
                top_gt = top_gt.reshape(14, 28)
                top_gt = top_gt.reshape(1, 14 * 28)
                top_proj = top_gt @ model.W_x.weight.t()
                bottom_est = top_proj @ model.V_y.weight.t()
                bottom_est = bottom_est.reshape(14, 28).detach().numpy()
                ax.imshow(bottom_est, cmap='gray', interpolation='none')
            elif plots[i] == 'retrieved bottom':
                bottom_ret = data[col_ind[img_idx]][1]
                bottom_ret = bottom_ret.reshape(14, 28)
                ax.imshow(bottom_ret, cmap='gray', interpolation='none')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()