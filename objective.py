import torch

def LTot(W_xX, W_yY, V_xW_xX, V_yW_yY, V_yW_xX, V_xW_yY, V_xW_yV_yW_xX, V_yW_xV_xW_yY, X, Y):
    loss1 = LG_c(W_xX, W_yY)
    loss2 = LG_x(V_xW_yY, X)
    loss3 = LG_y(V_yW_xX, Y)
    loss4 = LRec(V_xW_xX, V_yW_yY, X, Y)
    loss5 = LOrth(W_xX, W_yY) 
    loss6 = LCyc(V_xW_yV_yW_xX, V_yW_xV_xW_yY, X, Y)
    loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return loss4

def LG_c(W_xX, W_yY):
    loss = torch.nn.MSELoss(reduction='mean')
    loss = loss(W_xX, W_yY)
    loss = loss ** 2
    return loss

def LG_x(V_xW_yY, X):
    loss = torch.nn.MSELoss(reduction='mean')
    loss = loss(V_xW_yY, X)
    loss = loss ** 2
    return loss

def LG_y(V_yW_xX, Y):
    loss = torch.nn.MSELoss(reduction='mean')
    loss = loss(V_yW_xX, Y)
    loss = loss ** 2
    return loss

def LRec(V_xW_xX, V_yW_yY, X, Y):
    loss1 = torch.nn.MSELoss(reduction='mean')
    loss2 = torch.nn.MSELoss(reduction='mean')
    loss1 = loss1(V_xW_xX, X)
    loss2 = loss2(V_yW_yY, Y)
    loss1 = loss1 ** 2
    loss2 = loss2 ** 2
    return loss1 + loss2

def LOrth(W_xX, W_yY):
    loss1 = torch.nn.MSELoss(reduction='mean')
    loss2 = torch.nn.MSELoss(reduction='mean')
    temp1 = torch.matmul(W_xX, W_xX.t())
    temp2 = torch.matmul(W_yY, W_yY.t())    
    loss1 = loss1(temp1, torch.eye(temp1.shape[-2]))
    loss2 = loss2(temp2, torch.eye(temp2.shape[-2]))
    loss1 = loss1 ** 2
    loss2 = loss2 ** 2
    return loss1 + loss2    

def LCyc(V_xW_yV_yW_xX, V_yW_xV_xW_yY, X, Y):
    loss1 = torch.nn.MSELoss(reduction='mean')
    loss2 = torch.nn.MSELoss(reduction='mean')
    loss1 = loss1(V_xW_yV_yW_xX, X)
    loss2 = loss2(V_yW_xV_xW_yY, Y)
    loss1 = loss1 ** 2
    loss2 = loss2 ** 2
    return loss1 + loss2