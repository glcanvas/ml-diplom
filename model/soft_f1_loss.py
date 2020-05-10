import torch.nn as nn
import torch


class SoftF1Loss(nn.Module):

    def __init__(self):
        super(SoftF1Loss, self).__init__()

    def forward(self, output, target):
        """
        Calculate f measure with backward

        F1 = 2 * PRECISION * RECALL / (PRECISION + RECALL)
        PRECISION = TP / (TP + FP)
        RECALL = TP / (TP + FN)
        F1 = 2 * TP / (TP + FP) * TP / (TP + FN) / (TP / (TP + FP) + TP / (TP + FN))
        F1 = 2 * TP ^ 2 / ((TP + FP) * (TP + FN)) / ((TP * (TP + FN) + TP * (TP + FP)) / (TP + FN) * (TP + FP))
        =>
        2 * TP ^ 2                      2 * TP * TP
        -------------               =>
        ((TP + FP) * (TP + FN))         TP * TP + TP * FP + FN * TP + FN * FP
        -------------

        ((TP * (TP + FN) + TP * (TP + FP))      TP * TP + TP * FN + TP * TP + TP * FP
        -------------                       =>  -------------------------------------
        (TP + FN) * (TP + FP))                  TP * TP + TP * FP + FN * TP + FN * FP

        =>
        2 * TP * TP
        -------
        TP * TP + TP * FN + TP * TP + TP * FP
        =>
        --------------------
        | 2 * TP           |
        | ------           |
        | 2 * TP + FN + FP |
        --------------------

        :param output: model output batch_size x class_number (probability)
        :param target: trust output (0, or 1) batch_size x class_number
        :return:
        """
        true_positive = (output * target).sum(axis=0)
        false_positive = (output * (1 - target)).sum(axis=0)
        false_negative = ((1 - output) * target).sum(axis=0)
        soft_f1_measure = 2 * true_positive / (2 * true_positive + false_negative + false_positive + 1e-16)
        cost = 1 - soft_f1_measure
        return cost.sum()  # here not f1-macro measure, but just sum of f1-measures


if __name__ == "__main__":
    y = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    # y_m = torch.tensor([[0.5, 0.3, 0.8], [0.1, 0.6, 0.9]])
    y_m = torch.tensor([[1, 0.0, 1], [1, 1, 0.0]])

    loss = SoftF1Loss()
    print(loss.forward(y_m, y))
