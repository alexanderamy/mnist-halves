# MNIST Halves

This is an implementation of Supervised Correlation Analysis in Python with PyTorch for the purpose of matching the top halves of MNIST images with the corresponding bottom halves. It is based on the following paper:

Hoshen, Y.,  Wolf, L.,  "Unsupervised Correlation Analysis", CVPR, 2018, [arXiv:1804.00347](https://arxiv.org/pdf/1804.00347.pdf).

# Architecture

Supervised Correlation Analysis (SCA) is based on [Canonical Correlation Analysis](https://en.wikipedia.org/wiki/Canonical_correlation) (CCA), which seeks to model the relationships *between* two datasets in a way that is analogous to how PCA models the relationships between variables *within* a single dataset. 

To do this, we train matrices Wx and Wy to project datasets X and Y, respectively, onto a shared latent space C such that the sum of correlations between between the projected datasets are maximized, subject to projected the projected data from within a single dataset being uncorrelated. Matrices Vx and Vy are then trained to reconstruct vectors from C in the domain of the original data. The overall architecture is as follows:

![image](images/architecture.png)

# Objective Function
The loss minimzed during training comprises five main objectives:
1. Reconstruction of X in the domain of X / reconstruction of Y in the domain of Y (tops projected to tops)
2. Reconstruction of X in the domain of Y / reconstruction of Y in the domain of X (tops projected to bottoms)
3. Indistiguishability of projected datasets
4. Orthogonality of projected data 
5. Reconstruction of X in the domain of Y and back to X / reconstruction of Y in the domain of X and back to Y (tops projected to bottoms projected back to tops)

The individual loss functions are visualized below:
![image](images/reconstruction_loss_same.png)

![image](images/reconstruction_loss_opposite.png)

![image](images/indistinguishability_loss.png)

![image](images/orthogonality_loss.png)

![image](images/cycle_loss.png)

## Setup
1. `conda create -n mnist-halves python=3.8`
2. `conda activate mnist-halves` 
3. `pip install -r requirements.txt`
