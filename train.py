from objective import LTot

def train(model, optimizer, train_loader, test_loader, device, epochs=1):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        test_loss = 0

        for _, (X, Y) in enumerate(train_loader):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            loss = LTot(*model(X, Y))  
            loss.backward()
            optimizer.step()
            train_loss += loss.data

        model.eval()

        for _, (X, Y) in enumerate(test_loader):
            X, Y = X.to(device), Y.to(device)
            loss = LTot(*model(X, Y))  
            test_loss += loss.data

        train_loss = train_loss / len(train_loader)
        test_loss = test_loss / len(test_loader)
        print(f'epoch: {epoch} | train: {train_loss.item():.3f} | test: {test_loss.item():.3f}')
    
    return model