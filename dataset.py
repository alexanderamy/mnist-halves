from torch.utils.data import Dataset

class MNISTHalves(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        x = self.data_x[index]
        y = self.data_y[index]
        return x, y
    
    def __len__(self):
        return len(self.data_x)