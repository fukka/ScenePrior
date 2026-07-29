import torch
import torch.nn as nn
from .common import *

def fcn(num_input_channels=200, num_output_channels=1, num_hidden=1000):
    model = nn.Sequential()
    model.add(nn.Linear(num_input_channels, num_hidden, bias=True))
    model.add(nn.ReLU6())

    model.add(nn.Linear(num_hidden, num_output_channels))
    model.add(nn.Softmax())

    return model


class fcn_rgb_v1(nn.Module):
    def __init__(self, num_input_channels=200, num_output_channels=1, num_hidden=1000, RGB=False):
        super(fcn_rgb_v1, self).__init__()
        self.fc1 = nn.Linear(num_input_channels, num_hidden, bias=True)
        self.relu = nn.ReLU6()
        self.fc2 = nn.Linear(num_hidden, num_output_channels)
        self.softmax = nn.Softmax(dim=1)
        self.RGB = RGB

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        C = x.shape[0]
        if self.RGB:
            x = self.softmax(x.view(3, C // 3)).view(C)
        else:
            x = self.softmax(x.view(1, C)).view(C)
        return x

