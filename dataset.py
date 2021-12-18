from torch.utils.data import Dataset

class MNISTHalves(Dataset):
    def __init__(self, data_x, data_y, data_label):
        self.data_x = data_x
        self.data_y = data_y
        self.data_label = data_label

    def __getitem__(self, index):
        x = self.data_x[index]
        y = self.data_y[index]
        label = self.data_label[index]
        return x, y, label
    
    def __len__(self):
        return len(self.data_x)