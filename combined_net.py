from craft import CRAFT
from refinenet import RefineNet

import torch
import torch.nn as nn


class CombineNet(nn.Module):
    def __init__(self):
        super(CombineNet, self).__init__()

        self.base_net = CRAFT()
        self.refine_net = RefineNet()

    def forward(self, x):
        y, feature = self.base_net(x)
        y_refine = self.refine_net(y, feature)
        return y_refine
