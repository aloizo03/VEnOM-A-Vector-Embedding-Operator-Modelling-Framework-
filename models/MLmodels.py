import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, out_dim, num_of_layers=3, bias=False, init_=True, classification=True, device=None):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.layers_ = nn.Sequential()
        linear = nn.Linear(input_dim, hidden_dim, bias=bias,
                           device=self.device, dtype=torch.float64)
        if init_:
            nn.init.kaiming_uniform_(linear.weight, a=np.sqrt(5))
        self.layers_.add_module('linear_0', linear)
        self.layers_.add_module('activation_0', nn.ReLU())

        for i in range(num_of_layers - 1):
            linear = nn.Linear(hidden_dim, hidden_dim, bias=bias,
                               device=self.device, dtype=torch.float64)
            if init_:
                nn.init.kaiming_uniform_(linear.weight, a=np.sqrt(5))
            self.layers_.add_module(f'linear_{i + 1}', linear)
            self.layers_.add_module(f'activation_{i + 1}', nn.ReLU())

        linear = nn.Linear(hidden_dim, out_dim, bias=bias,
                           device=self.device, dtype=torch.float64)
        if init_:
            nn.init.kaiming_uniform_(linear.weight, a=np.sqrt(5))
        self.layers_.add_module('out_layer', linear)
        if classification:
            self.layers_.add_module('Sigmoid', nn.Sigmoid())

        # self.softmax_ = F.softmax

    def forward(self, x):
        out = self.layers_(x)
        return out
