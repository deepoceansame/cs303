from models.YourNet import YourNet2
from models.YourNet_discard import YourNet
import argparse
import torch

from torchvision import datasets, transforms
from eval.metrics import get_accuracy, get_infer_time, get_macs_and_params
model = YourNet()
MACs, params = get_macs_and_params(model, 'cpu')
print(MACs / (1000 ** 2), params / (1000 ** 2))

