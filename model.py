import torch.nn as nn

IMG_DIM = 14 * 28
LATENT_DIM = 10

# Supervised UCA - source: https://arxiv.org/pdf/1804.00347.pdf
class SupUCA(nn.Module):
    def __init__(self, img_dim=IMG_DIM, latent_dim=LATENT_DIM):
        super(SupUCA, self).__init__()
        self.img_dim = img_dim
        self.latent_dim = latent_dim
        self.W_x = nn.Linear(self.img_dim, self.latent_dim, bias=False)
        self.W_y = nn.Linear(self.img_dim, self.latent_dim, bias=False)
        self.V_x = nn.Linear(self.latent_dim, self.img_dim, bias=False)
        self.V_y = nn.Linear(self.latent_dim, self.img_dim, bias=False)

    def forward(self, X, Y):
        batch_dim = X.shape[0]
        
        # CCA takes as input sets of matching views X_i and Y_i, which
        # are stacked as the matching columns of two matrices X and Y
        X = X.view(batch_dim, self.img_dim)
        Y = Y.view(batch_dim, self.img_dim)

        # project X and Y onto shared latent space via W_x and W_y, respectively
        W_xX = self.W_x(X)
        W_yY = self.W_y(Y)

        # reconstruct projections in domain of original input
        V_xW_xX = self.V_x(W_xX)
        V_yW_yY = self.V_y(W_yY)

        # reconstruct projections in domains of opposite input
        V_yW_xX = self.V_y(W_xX)
        V_xW_yY = self.V_x(W_yY)
        
        # project X and Y onto shared latent space via W_x and W_y, respectively
        # reconstruct projections in domains of opposite input
        # project reconstructed X and Y back onto shared latent space, this time via W_y and W_x, respectively
        # reconstruct projections in domains of original input
        V_xW_yV_yW_xX = self.V_x(self.W_y(V_yW_xX))
        V_yW_xV_xW_yY = self.V_x(self.W_x(V_xW_yY))

        return W_xX, W_yY, V_xW_xX, V_yW_yY, V_yW_xX, V_xW_yY, V_xW_yV_yW_xX, V_yW_xV_xW_yY, X, Y