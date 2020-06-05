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


class ConvConnectionProductBlock(nn.Module):

    def __init__(self, base_network_layers: int, classes: int):
        super(ConvConnectionProductBlock, self).__init__()
        self.base_network_layers = base_network_layers
        self.classes = classes
        self.lowering_layer = nn.Conv2d(base_network_layers, classes, (1, 1))

    def forward(self, am_out, first_out):
        first_out_1 = self.lowering_layer(first_out)
        return first_out_1 * am_out


class ConvConnectionSumBlock(nn.Module):

    def __init__(self, base_network_layers: int, classes: int):
        super(ConvConnectionSumBlock, self).__init__()
        self.base_network_layers = base_network_layers
        self.classes = classes
        self.lowering_layer = nn.Conv2d(base_network_layers, classes, (1, 1))

    def forward(self, am_out, first_out):
        first_out_1 = self.lowering_layer(first_out)
        return first_out_1 + am_out


def determine_output_size(model_sublayer: nn.Module) -> int:
    image = torch.ones((1, 3, 224, 224))
    out = model_sublayer(image)
    print(out.shape)
    return out.shape[1]
