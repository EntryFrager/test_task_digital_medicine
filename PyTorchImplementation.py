import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
import gzip


def load_image(filename):
    # Read the inputs in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # The inputs are vectors now, we reshape them to monochrome 2D images
    data = data.reshape(-1, 28, 28)
    # The inputs come as bytes, we convert them to float32 in range [0,1].
    return (data / np.float32(256)).squeeze()


def load_mnist_labels(filename):
    # Read the labels in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    # The labels are vectors of integers now, that's exactly what we want.
    return data


class MNISTDataset(Dataset):
    def __init__(self):



class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512, momentum = 0.1),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, momentum = 0.1),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 10),
            nn.LogSoftmax(dim=1)
        )


    def forward(self, x):
        return self.layers(x)


X_train = load_image('data/train-images-idx3-ubyte.gz')
X_test = load_image('data/t10k-images-idx3-ubyte.gz')
Y_train = load_mnist_labels('data/train-labels-idx1-ubyte.gz')
Y_test = load_mnist_labels('data/t10k-labels-idx1-ubyte.gz')
X_train, X_val = X_train[:-10000], X_train[-10000:]
Y_train, Y_val = Y_train[:-10000], Y_train[-10000:]

model = MNISTNet()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
