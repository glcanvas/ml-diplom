import torch.nn as nn
import torch
import math
import torch.nn.functional as F

EPS = 1e-15

# losses

# based on https://arxiv.org/pdf/1708.02002.pdf

class FocalLoss(nn.Module):

    def __init__(self, alpha: float, gamma: float):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.alpha_list = None

    def apply_alpha_list(self, count_classes: torch.Tensor, dataset_size: int):
        #
        # frequency of positive labels
        self.alpha_list = count_classes / dataset_size

    def forward(self, output, target):
        if self.alpha_list is None:
            return self._forward_single_alpha(output, target)
        else:
            return self._forward_alpha_list(output, target)

    def _forward_alpha_list(self, output, target):
        # target = 0 or 1 vector
        # output = probability vector [0, 1]
        inverse_target = 1 - target
        inverse_output = 1 - output

        # equivalent
        # p if y == 1
        # 1 - p if y == 0
        p_t = (output * target + inverse_target * inverse_output) + EPS

        # equivalent
        # -log(p), if y = 1
        # -log(1 - p), if y = 0
        # cross_entropy -- matrix: batch_size x classes
        cross_entropy = -1 * torch.log(p_t)

        # inverse class frequency
        # alpha_list -- frequency for 1 values in target
        # alpha -- frequency for 0 values in target
        # inverse_alpha -- frequency for 1 values in target
        alpha = (1 - self.alpha_list) * target
        inverse_alpha = self.alpha_list * inverse_target
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

    def _forward_single_alpha(self, output, target):
        # target = 0 or 1 vector
        # output = probability vector [0, 1]
        inverse_target = 1 - target
        inverse_output = 1 - output

        # equivalent
        # p if y == 1
        # 1 - p if y == 0
        p_t = (output * target + inverse_target * inverse_output) + EPS

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
    y_m = torch.tensor([[0.000000000001, 0.3, 0.8], [0.1, 0.6, 0.9]], requires_grad=True)

    loss = FocalLoss(1, 2)
    loss_v = loss.forward(y_m, y)
    print(loss_v)
    loss_v.backward()
    print(y_m.grad)
