import torch.nn as nn
from utils import property as p


class SoftF1Loss(nn.Module):

    def __init__(self):
        super(SoftF1Loss, self).__init__()

    def forward(self, output, target):
        """
        Calculate f measure with backward
        :param output: model output batch_size x class_number (probability)
        :param target: trust output (0, or 1) batch_size x class_number
        :return:
        """
        pass
