import torch
import torch.nn as nn
import torch.nn.functional as F

class SkeletonMLP(nn.Module):
    def __init__(self):
        super(SkeletonMLP, self).__init__()
        self.fc1 = nn.Linear(33, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # binary classifier: mannequin or background

    def forward(self, x):  # x: [B, 33]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
