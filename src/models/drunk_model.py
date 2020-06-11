import torch
import torch.nn as nn
import torch.nn.functional as F

class DrunkNet(nn.Module):
    def __init__(self):
        super(DrunkNet, self).__init__()
        self.fc1 = nn.Linear(120*4096, 32)
        self.fc4 = nn.Linear(32, 2)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 120*4096)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        return x
