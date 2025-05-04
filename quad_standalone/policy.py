import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(num_inputs, 512)
        self.layer_norm = nn.LayerNorm(512, eps=1e-6)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer_norm(x)
        x = nn.functional.elu(x)
        x = self.layer2(x)
        x = nn.functional.elu(x)
        x = self.layer3(x)
        x = nn.functional.elu(x)
        x = self.layer4(x)
        x = torch.clamp(x, -10, 10)
        return x
