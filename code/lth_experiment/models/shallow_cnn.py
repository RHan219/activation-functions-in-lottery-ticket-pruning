import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowCNN(nn.Module):
    def __init__(self, activation_fn=nn.ReLU()):
        super().__init__()

        self.act = activation_fn

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = self.act(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x
