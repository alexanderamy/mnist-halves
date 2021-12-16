import torch

def LG_c(W_xX, W_yY):
    loss = torch.nn.MSELoss(reduction='mean')
    loss = loss(W_xX, W_yY)
    # loss = loss ** 2
    return loss

def LG_x(V_xW_yY, X):
    loss = torch.nn.MSELoss(reduction='mean')
    loss = loss(V_xW_yY, X)
    # loss = loss ** 2
    return loss

def LG_y(V_yW_xX, Y):
    loss = torch.nn.MSELoss(reduction='mean')
    loss = loss(V_yW_xX, Y)
    # loss = loss ** 2
    return loss

def LRec(V_xW_xX, V_yW_yY, X, Y):
    loss1 = torch.nn.MSELoss(reduction='mean')
    loss2 = torch.nn.MSELoss(reduction='mean')
    loss1 = loss1(V_xW_xX, X)
    loss2 = loss2(V_yW_yY, Y)
    # loss1 = loss1 ** 2
    # loss2 = loss2 ** 2
    return loss1 + loss2

# def LOrth(W_xX, W_yY):
#     temp1 = torch.matmul(W_xX.t(), W_xX)
#     temp2 = torch.matmul(W_yY.t(), W_yY)
#     temp1 = temp1 - torch.eye(temp1.shape[-2])
#     temp2 = temp2 - torch.eye(temp2.shape[-2])
#     # loss1 = torch.linalg.matrix_norm(temp1)
#     # loss2 = torch.linalg.matrix_norm(temp2)
#     loss1 = torch.linalg.matrix_norm(torch.matmul(W_xX.t(), W_xX) - torch.eye(torch.matmul(W_xX.t(), W_xX).shape[-2]))
#     loss2 = torch.linalg.matrix_norm(torch.matmul(W_yY.t(), W_yY) - torch.eye(torch.matmul(W_yY.t(), W_yY).shape[-2]))

#     # loss1 = loss1 ** 2
#     # loss2 = loss2 ** 2
#     return loss1 + loss2

# def LOrth(W_xX, W_yY):
#     loss1 = torch.nn.MSELoss(reduction='mean')
#     loss2 = torch.nn.MSELoss(reduction='mean')
    
#     temp1 = torch.matmul(W_xX.t(), W_xX)
#     temp2 = torch.matmul(W_yY.t(), W_yY)

#     # temp1 = temp1 - torch.eye(temp1.shape[-2])
#     # temp2 = temp2 - torch.eye(temp2.shape[-2])
    
#     loss1 = loss1(temp1, torch.eye(temp1.shape[-2]))
#     loss2 = loss2(temp2, torch.eye(temp2.shape[-2]))
#     # loss1 = loss1 ** 2
#     # loss2 = loss2 ** 2
#     return loss1 + loss2

def LOrth(W_xX, W_yY):
    W_xX = W_xX.view(W_xX.shape[0], -1)
    W_yY = W_yY.view(W_yY.shape[0], -1)
    loss1 = torch.matmul(torch.t(W_xX), W_xX)
    loss2 = torch.matmul(torch.t(W_yY), W_yY)
    loss1 = loss1 - torch.eye(W_xX.shape[1])
    loss2 = loss2 - torch.eye(W_yY.shape[1])
    return loss1.abs().sum() + loss2.abs().sum()
    

    param_flat = param.view(param.shape[0], -1)
    sym = torch.mm(param_flat, torch.t(param_flat))
    sym -= torch.eye(param_flat.shape[0])
    orth_loss = orth_loss + (reg * sym.abs().sum())

def LCyc(V_xW_yV_yW_xX, V_yW_xV_xW_yY, X, Y):
    loss1 = torch.nn.MSELoss(reduction='mean')
    loss2 = torch.nn.MSELoss(reduction='mean')
    loss1 = loss1(V_xW_yV_yW_xX, X)
    loss2 = loss2(V_yW_xV_xW_yY, Y)
    # loss1 = loss1 ** 2
    # loss2 = loss2 ** 2
    return loss1 + loss2