import torch
import torch.nn as nn


class ffNet(nn.Module):
    def __init__(self, dim, hidden_size=100, hidden_layer=1):
        super(ffNet, self).__init__()
        self.d = dim
        self.m = hidden_size
        self.L = hidden_layer
        self.modules = [nn.Linear(dim, hidden_size),nn.ReLU()]
        for i in range (1, hidden_layer-1):
            self.modules.append(nn.Linear(hidden_size, hidden_size))
            self.modules.append(nn.ReLU())
            
        self.modules.append(nn.Linear(hidden_size, 1))
        # self.modules.append(nn.ReLU())
        self.sequential = nn.ModuleList(self.modules)

    def forward(self, x):
        x = x.float()
        for layer in self.sequential: 
            x = layer(x)
        return x