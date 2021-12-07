# load model
import torch
import os
from torchvision import datasets, transforms
from models.LeNet5 import LeNet5
from eval.metrics import get_accuracy, get_infer_time, get_macs_and_params
import torch.nn.utils.prune

device = 'cpu'

model = LeNet5().to(device=device)
model.load_state_dict(torch.load('./checkpoints/LeNet5/epoch-6.pth', map_location=device))

# Inspect parameters of conv1 of the model
# named_parameters and named_buffers
conv1_module = model.conv1
print('named_parameters')
print(list(conv1_module.named_parameters()))
print('named_buffers')
print(list(conv1_module.named_buffers))


#
