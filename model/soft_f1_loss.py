import torch.nn as nn
import torch
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
        true_positive = (output * target).sum(axis=0)
        false_positive = (output * (1 - target)).sum(axis=0)


if __name__ == "__main__":
    y = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    y_m = torch.tensor([[0.5, 0.3, 0.8], [0.1, 0.6, 0.9]])

    loss = SoftF1Loss()
    loss.forward(y_m, y)
