import torch
import torch.nn as nn
import math
import property as p

class AmLossFunction(nn.Module):
    """
    loss function 1 / x - 1, for non null cell, for other BCEloss
    """

    def __init__(self, zero_cell_loss=nn.BCELoss()):
        super(AmLossFunction, self).__init__()
        self.zero_cell_loss = zero_cell_loss

    def forward(self, model_output, segment):
        # segments
        ones = model_output * segment
        segment_loss = self.__custom_loss(ones, segment).sum()
        # non segments cells
        zeros = model_output * (1 - segment)
        non_segment_loss = self.zero_cell_loss(zeros, (1 - segment))
        return segment_loss + non_segment_loss

    def __custom_loss(self, ones, segment):
        v = segment * (1 / (ones + p.EPS) - 1)
        # v[v == math.inf] = 0
        # v[v != v] = 0
        return v
