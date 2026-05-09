import torch
import torch.nn as nn

class GeoConvNet(nn.Module):
    def __init__(self, input_dim):
        super(GeoConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(32 * input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x