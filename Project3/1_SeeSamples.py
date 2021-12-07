import torch
from torch import nn
import os
from torchvision import datasets, transforms
import torch.nn.functional as F
from models.LeNet5 import LeNet5
from eval.metrics import get_accuracy, get_infer_time, get_macs_and_params

batch_size = 4
device = 'cuda'
conv1 = nn.Conv2d(1, 6, 3).to(device)
conv2 = nn.Conv2d(6, 16, 3).to(device)
train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                           ])),
        batch_size=batch_size, shuffle=True)
for X, y in train_loader:
    X, y = X.to(device), y.to(device)
    X = F.max_pool2d(F.relu(conv1(X)), (2, 2))
    X = F.max_pool2d(F.relu(conv2(X)), 2)
    X = X.view(-1, int(X.nelement() / X.shape[0]))
    print(X.shape)
    break

'''
image_size: 28 * 28
'''
