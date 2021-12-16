import torch
import torch.optim as optim
from tqdm import tqdm
from train import train
from model import SupUCA
from dataset import MNISTHalves
from torchvision import datasets
from torch.utils.data import DataLoader
from utils import split_img, ToPILImage, ToTensor

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load training data
train_dataset = datasets.MNIST(root='../data/', train=True, download=True)
train_x = []
train_y = []

for i in tqdm(range(train_dataset.data.shape[0])):
    img = ToPILImage(train_dataset.data[i])
    x, y = split_img(img)
    train_x.append(ToTensor(x))
    train_y.append(ToTensor(y))

train_x = torch.stack(train_x)
train_y = torch.stack(train_y)

# load test data
test_dataset = datasets.MNIST(root='../data/', train=False, download=True)
test_x = []
test_y = []

for i in tqdm(range(test_dataset.data.shape[0])):
    img = ToPILImage(test_dataset.data[i])
    x, y = split_img(img)
    test_x.append(ToTensor(x))
    test_y.append(ToTensor(y))

test_x = torch.stack(test_x)
test_y = torch.stack(test_y)

# build datasets / data loaders
batch_size = 128
train_data = MNISTHalves(train_x, train_y)
test_data = MNISTHalves(test_x, test_y)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# build model
model = SupUCA()
model.to(device)

# set optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)

# train model
train(model, optimizer, train_loader, test_loader, device, epochs=1)

