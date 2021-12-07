from torch import nn
import torch.nn.functional as F


class YourNet(nn.Module):
    def __init__(self):
        super(YourNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.conv2 = nn.Conv2d(3, 5, 3)
        self.fc1 = nn.Linear(5 * 5 * 5, 75)  # 5x5 image dimension
        self.fc2 = nn.Linear(75, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
