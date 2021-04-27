import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.initializer import init_rnn_wt, init_linear_wt


class ReduceState(nn.Module):
    def __init__(self, H1, H2):
        super(ReduceState, self).__init__()
        self.W = nn.Linear(H1, H2)
        init_linear_wt(self.W)

    def forward(self, h):
        return self.W(h)
