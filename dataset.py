from torch.utils.data import Dataset

class MNISTHalves(Dataset):
    def __init__(self, tops, bottoms, labels):
        self.tops = tops
        self.bottoms = bottoms
        self.labels = labels

    def __getitem__(self, index):
        x = self.tops[index]
        y = self.bottoms[index]
        label = self.labels[index]
        return x, y, label
    
    def __len__(self):
        return len(self.tops)