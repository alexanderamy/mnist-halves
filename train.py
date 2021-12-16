from objective import (
    LG_c,
    LG_x,
    LG_y,
    LRec,
    LOrth,
    LCyc
)

def train(model, optimizer, train_loader, test_loader, device, epochs=1):
    model.train()
    train_loss = 0
    
    for batch_idx, (X, Y) in enumerate(train_loader):
        X, Y = X.to(device), Y.to(device)

        optimizer.zero_grad()
        W_xX, W_yY, V_xW_xX, V_yW_yY, V_yW_xX, V_xW_yY, V_xW_yV_yW_xX, V_yW_xV_xW_yY, X, Y = model(X, Y)
        
        l1 = LG_c(W_xX, W_yY)
        l2 = LG_x(V_xW_yY, X)
        l3 = LG_y(V_yW_xX, Y)
        l4 = LRec(V_xW_xX, V_yW_yY, X, Y)
        l5 = LOrth(W_xX, W_yY) 
        l6 = LCyc(V_xW_yV_yW_xX, V_yW_xV_xW_yY, X, Y)
        print(batch_idx, l1.item(), l2.item(), l3.item(), l4.item(), l5.item(), l6.item())

        loss = l1 + l2 + l3 + l4 + l5 + l6
        # loss = l1 + l2 + l3 + l4 + l6
        # loss = l5
               
        loss.backward()
        optimizer.step()

        train_loss += loss.data
        # if batch_idx % 10 == 0:
        #     print(f'Epoch: {epoch} | Batch: {batch_idx} |  Loss: ({train_loss/(batch_idx+1):.4f})')
        