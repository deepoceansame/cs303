import argparse
import os
import torch
from torch import nn
from torchvision import datasets, transforms
from models.LeNet5 import LeNet5
from models.YourNet import YourNet
from eval.metrics import get_accuracy

loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1, shuffle=False)
i = 0
for x, y in loader:
    # print(texts)
    print(y)
    i += 1
    if i>5:
        break

i = 0
for x, y in loader:
    # print(texts)
    print(y)
    i += 1
    if i>5:
        break