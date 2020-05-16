import torch.nn as nn
import torch
import math
import torch.nn.functional as F


# losses

# based on https://arxiv.org/pdf/1708.02002.pdf

class FocalLoss(nn.Module):

    def __init__(self, alpha: float, gamma: float):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, output, target):
        # target = 0 or 1 vector
        # output = probability vector [0, 1]
        inverse_target = 1 - target
        inverse_output = 1 - output

        # equivalent
        # p if y == 1
        # 1 - p if y == 0
        p_t = (output * target + inverse_target * inverse_output)

        # equivalent
        # -log(p), if y = 1
        # -log(1 - p), if y = 0
        # cross_entropy -- matrix: batch_size x classes
        cross_entropy = -1 * torch.log(p_t)

        alpha = self.alpha * target
        inverse_alpha = (1 - self.alpha) * inverse_target
        # equivalent
        # a if y == 1
        # 1 - a if y == 0
        alpha_t = alpha + inverse_alpha

        # - log(p_t) == cross_entropy
        # a_t == alpha_t
        # (1 - p_t) ** y == (1 - p_t).pow(self.gamma)
        # - a_t * (1 - p_t) ** y * log(p_t)
        focal_loss = alpha_t * (1 - p_t).pow(self.gamma) * cross_entropy

        return focal_loss.sum()


if __name__ == "__main__":
    y = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    # y_m = torch.tensor([[1, 0.0, 1], [1, 1, 0.0]])
    y_m = torch.tensor([[0.5, 0.3, 0.8], [0.1, 0.6, 0.9]], requires_grad=True)

    loss = FocalLoss(1, 2)
    loss_v = loss.forward(y_m, y)
    print(loss_v)
    loss_v.backward()
    print(y_m.grad)