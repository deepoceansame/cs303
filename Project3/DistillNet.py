import copy
import math
import torch.nn.functional as F
import torch.nn as nn
import torch
from models.LeNet5 import LeNet5
from models.YourNet import YourNet2

class DistillNet(nn.Module):

    def __init__(self, t_net, s_net):
        super(DistillNet, self).__init__()

        self.t_net = t_net
        self.s_net = s_net
        self.out_t = None
        self.out_s = None

    def forward(self, x):

        t_net = self.t_net
        s_net = self.s_net

        # Teacher network
        out_t = t_net(x)
        self.out_t = out_t
        # Student network
        out_s = s_net(x)
        self.out_s = out_s

        return out_s

device = 'cpu'
t_net = LeNet5().to(device=device)
t_net.load_state_dict(torch.load('./checkpoints/LeNet5/epoch-6.pth', map_location=device))

s_net = YourNet2().to(device=device)
d_net = DistillNet(t_net, s_net).to(device=device)

