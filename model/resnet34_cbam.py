import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as m


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class ExtendedBasicBlock(nn.Module):
    def __init__(self,
                 conv1: nn.Module,
                 bn1: nn.Module,
                 relu: nn.Module,
                 conv2: nn.Module,
                 bn2: nn.Module,
                 downsample: nn.Module,
                 stride: nn.Module,
                 ):
        super(ExtendedBasicBlock, self).__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = relu
        self.conv2 = conv2
        self.bn2 = bn2
        self.downsample = downsample
        self.stride = stride
        self.cbam = CBAM(self.bn2.num_features, 16)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xm: tuple) -> tuple:
        x = xm[0]
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        cbam_out = self.cbam(out)

        out = cbam_out + residual
        out = self.relu(out)
        xm[1].append(cbam_out)
        return out, xm[1]


def extend_layer(layer: nn.Module) -> nn.Module:
    res = []
    for l in layer:
        res.append(ExtendedBasicBlock(l.conv1,
                                      l.bn1,
                                      l.relu,
                                      l.conv2,
                                      l.bn2,
                                      l.downsample,
                                      l.stride))
    return nn.Sequential(*res)


def build_resnet34_with_cbam(classes: int):
    model = m.resnet34(pretrained=True)
    conv1 = model.conv1
    bn1 = model.bn1
    relu = model.relu
    maxpool = model.maxpool
    layer1 = extend_layer(model.layer1)
    layer2 = extend_layer(model.layer2)
    layer3 = extend_layer(model.layer3)
    layer4 = extend_layer(model.layer4)
    avgpool = model.avgpool
    fc = nn.Linear(512, classes)

    puller_list = []
    for i in range(3):
        module = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), nn.MaxPool2d(kernel_size=2, stride=2))
        puller_list.append(module)
    for i in range(4):
        module = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), nn.MaxPool2d(kernel_size=4, stride=4))
        puller_list.append(module)
    for i in range(6):
        module = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), nn.MaxPool2d(kernel_size=8, stride=8))
        puller_list.append(module)
    for i in range(3):
        module = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), nn.MaxPool2d(kernel_size=15, stride=15))
        puller_list.append(module)

    return ModelWithBAM(conv1,
                        bn1,
                        relu,
                        maxpool,
                        layer1,
                        layer2,
                        layer3,
                        layer4,
                        avgpool,
                        fc,
                        classes), puller_list


class ModelWithBAM(nn.Module):

    def __init__(self,
                 conv1: nn.Module,
                 bn1: nn.Module,
                 relu: nn.Module,
                 maxpool: nn.Module,
                 layer1: nn.Module,
                 layer2: nn.Module,
                 layer3: nn.Module,
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
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.avgpool = avgpool
        self.fc = fc
        self.sigmoid = nn.Sigmoid()

        self.flatten = []
        for i in range(3):
            module = nn.Conv2d(64, classes, (1, 1))
            self.add_module("flat-{}-{}".format(64, i), module)
            self.flatten.append(module)
        for i in range(4):
            module = nn.Conv2d(128, classes, (1, 1))
            self.add_module("flat-{}-{}".format(128, i), module)
            self.flatten.append(module)
        for i in range(6):
            module = nn.Conv2d(256, classes, (1, 1))
            self.add_module("flat-{}-{}".format(256, i), module)
            self.flatten.append(module)
        for i in range(3):
            module = nn.Conv2d(512, classes, (1, 1))
            self.add_module("flat-{}-{}".format(512, i), module)
            self.flatten.append(module)

    def forward(self, x: torch.Tensor) -> tuple:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        masks = []
        x, masks = self.layer1((x, masks))
        x, masks = self.layer2((x, masks))
        x, masks = self.layer3((x, masks))
        x, masks = self.layer4((x, masks))

        x = self.avgpool(x)
        x = self.fc(torch.flatten(x, 1))

        masks = list(map(lambda x: self.sigmoid(x[0](x[1])), zip(self.flatten, masks)))
        return self.sigmoid(x), masks


if __name__ == "__main__":
    model, puller = build_resnet34_with_cbam(5)
    print(model)
    print(puller)
    image = torch.ones((4, 3, 224, 224))
    segments = torch.ones((4, 5, 224, 224))
    label = torch.ones((4, 5))
    x, masks = model(image)
    loss = nn.BCELoss()
    c, sl = model(image)
    cl_loss = loss(x, label)
    print(cl_loss)
    cl_loss.backward(retain_graph=True)
    segment_loss = []
    for p, s in zip(puller, sl):
        l = loss(s, p(segments))
        l.backward(retain_graph=True)
        segment_loss.append(l)
    print(segment_loss)
"""
torch.Size([4, 5, 56, 56])
torch.Size([4, 5, 56, 56])
torch.Size([4, 5, 56, 56])
torch.Size([4, 5, 28, 28])
torch.Size([4, 5, 28, 28])
torch.Size([4, 5, 28, 28])
torch.Size([4, 5, 28, 28])
torch.Size([4, 5, 14, 14])
torch.Size([4, 5, 14, 14])
torch.Size([4, 5, 14, 14])
torch.Size([4, 5, 14, 14])
torch.Size([4, 5, 14, 14])
torch.Size([4, 5, 14, 14])
torch.Size([4, 5, 7, 7])
torch.Size([4, 5, 7, 7])
torch.Size([4, 5, 7, 7])
"""
