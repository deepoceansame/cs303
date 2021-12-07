import torch
import torch.nn as nn
from torchvision import datasets, transforms
from models.LeNet5 import LeNet5
from eval.metrics import get_accuracy, get_infer_time, get_macs_and_params

device = 'cpu'
model = LeNet5().to(device=device)
model.load_state_dict(torch.load('./checkpoints/LeNet5/epoch-6.pth', map_location=device))

print(model)
total = 0

percent = 0.4
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        total += m.weight.data.shape[0]
        print(m.weight)
print(total)
bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        size = m.weight.data.shape[0]
        print(size, index)
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size

y, i = torch.sort(bn)
thre_index = int(total * percent)
thre = y[thre_index]

print('thre:', thre)

pruned = 0
cfg = []
cfg_mask = []
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.Conv2d):
        weight_copy = m.weight.data.clone()
        mask = weight_copy.abs().gt(thre).float().cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

pruned_ratio = pruned/total
print(pruned_ratio)
print(cfg)