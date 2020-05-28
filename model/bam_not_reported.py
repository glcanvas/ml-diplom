import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.models as m


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module('gate_c_fc_%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_bn_%d' % (i + 1), nn.BatchNorm1d(gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_relu_%d' % (i + 1), nn.ReLU())
        self.gate_c.add_module('gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d(in_tensor, in_tensor.size(2), stride=in_tensor.size(2))
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)


class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module('gate_s_conv_reduce0',
                               nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0', nn.BatchNorm2d(gate_channel // reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module('gate_s_conv_di_%d' % i,
                                   nn.Conv2d(gate_channel // reduction_ratio, gate_channel // reduction_ratio,
                                             kernel_size=3, \
                                             padding=dilation_val, dilation=dilation_val))
            self.gate_s.add_module('gate_s_bn_di_%d' % i, nn.BatchNorm2d(gate_channel // reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1))

    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)


class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)

    def forward(self, in_tensor):
        att = 1 + F.sigmoid(self.channel_att(in_tensor) * self.spatial_att(in_tensor))
        return att * in_tensor


def build_model_with_bam(classes: int, model: nn.Module):
    conv1 = model.conv1
    bn1 = model.bn1
    relu = model.relu
    maxpool = model.maxpool
    layer1 = model.layer1
    bam1 = BAM(64)
    layer2 = model.layer2
    bam2 = BAM(128)
    layer3 = model.layer3
    bam3 = BAM(256)
    layer4 = model.layer4
    avgpool = model.avgpool
    fc = nn.Linear(512, classes)
    return ModelWithBAM(conv1,
                        bn1,
                        relu,
                        maxpool,
                        layer1,
                        bam1,
                        layer2,
                        bam2,
                        layer3,
                        bam3,
                        layer4,
                        avgpool,
                        fc,
                        classes)


class ModelWithBAM(nn.Module):

    def __init__(self,
                 conv1: nn.Module,
                 bn1: nn.Module,
                 relu: nn.Module,
                 maxpool: nn.Module,
                 layer1: nn.Module,
                 bam1: nn.Module,
                 layer2: nn.Module,
                 bam2: nn.Module,
                 layer3: nn.Module,
                 bam3: nn.Module,
                 layer4: nn.Module,
                 avgpool: nn.Module,
                 fc: nn.Module,
                 classes: int
                 ):
        super(ModelWithBAM, self).__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = relu
        self.maxpool = maxpool
        self.layer1 = layer1
        self.bam1 = bam1
        self.layer2 = layer2
        self.bam2 = bam2
        self.layer3 = layer3
        self.bam3 = bam3
        self.layer4 = layer4
        self.avgpool = avgpool
        self.fc = fc

        self.flatten1 = nn.Conv2d(64, classes, (1, 1))
        self.flatten2 = nn.Conv2d(128, classes, (1, 1))
        self.flatten3 = nn.Conv2d(256, classes, (1, 1))

    def forward(self, x: torch.Tensor) -> tuple:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        x_bam1 = self.bam1(x)
        x = self.layer2(x_bam1)
        x_bam1 = self.flatten1(x_bam1)

        x_bam2 = self.bam2(x)
        x = self.layer3(x_bam2)
        x_bam2 = self.flatten2(x_bam2)

        x_bam3 = self.bam3(x)
        x = self.layer4(x_bam3)
        x_bam3 = self.flatten3(x_bam3)

        x = self.avgpool(x)
        x = self.fc(torch.flatten(x, 1))
        return x, [x_bam1, x_bam2, x_bam3]


if __name__ == "__main__":
    model = build_model_with_bam(5, m.resnet18(pretrained=True))
    image = torch.ones((4, 3, 224, 224))
    x, segments = model(image)
    print(x)
