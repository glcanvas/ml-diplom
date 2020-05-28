import torch
import torch.nn as nn


class ConnectionProductBlock(nn.Module):

    def __init__(self):
        super(ConnectionProductBlock, self).__init__()

    def forward(self, am_out, first_out):
        point_wise_out = None

        for c in range(0, am_out.size(1)):
            sub_am_out = am_out[:, c:c + 1, :, :]
            if point_wise_out is None:
                point_wise_out = first_out * sub_am_out
            else:
                point_wise_out = torch.cat((point_wise_out, first_out * sub_am_out), dim=1)
        return point_wise_out


class ConnectionSumBlock(nn.Module):

    def __init__(self):
        super(ConnectionSumBlock, self).__init__()

    def forward(self, am_out, first_out):
        point_wise_out = None

        for c in range(0, am_out.size(1)):
            sub_am_out = am_out[:, c:c + 1, :, :]
            if point_wise_out is None:
                point_wise_out = first_out + sub_am_out
            else:
                point_wise_out = torch.cat((point_wise_out, first_out + sub_am_out), dim=1)
        return point_wise_out
