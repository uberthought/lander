import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim, momentum=0.01)
        self.act1 = nn.LeakyReLU()

        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim, momentum=0.01)
        self.act2 = nn.LeakyReLU()

    def forward(self, x):
        residual = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.fc2(out)
        out = self.bn2(out)

        # Skip connection
        out = out + residual

        # Final activation for this block
        out = self.act2(out)
        return out