import torch
import torch.nn as nn
import torchvision.models as m
import torch.nn.functional as F

import sys

sys.path.insert(0, "/home/nduginec/ml3/ml-diplom")
sys.path.insert(0, "/home/ubuntu/ml3/ml-diplom")

import model.connection_block as cb

"""
ONLY RESNET18
RESNET34!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUPPORT OTHER TOO LAZY!!!!!!!!
"""
"""
Давайте разобьем модель на 4 + части
1 -- та что до бранчевания
2 -- левай ветка
3 -- правая ветка
4 -- их соедниение + все остальное
"""


def build_attention_module_model(classes: int, resnet_model: nn.Module, connection_block: nn.Module, pretrained=True):
    model = resnet_model

    conv1 = model.conv1
    bn1 = model.bn1
    relu = model.relu
    maxpool = model.maxpool
    layer1 = model.layer1
    layer2 = model.layer2
    layer3 = model.layer3
    layer4 = model.layer4
    avgpool = model.avgpool
    fc = nn.Linear(512, classes)

    basis_branch = nn.Sequential(
        conv1,
        bn1,
        relu,
        maxpool,
        layer1
    )

    first_branch = nn.Sequential(
        layer2,
        layer3
    )

    if isinstance(connection_block, cb.ConnectionProductBlock):
        am_branch = nn.Sequential(
            *m.vgg16(pretrained=pretrained).features[2:15],
            nn.Conv2d(256, classes, kernel_size=(3, 3), padding=(1, 1)),
            nn.Sigmoid()
        )
        sigmoid_out_flag = False
    else:
        am_branch = nn.Sequential(
            *m.vgg16(pretrained=pretrained).features[2:15],
            nn.Conv2d(256, classes, kernel_size=(3, 3), padding=(1, 1))
        )
        sigmoid_out_flag = True

    merged_branch = nn.Sequential(
        nn.Conv2d(256 * classes, 256, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
        layer4,
        avgpool
    )
    puller = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), nn.MaxPool2d(kernel_size=2, stride=2),
                           nn.MaxPool2d(kernel_size=4, stride=4))

    sam_model = AttentionModuleModel(basis_branch,
                                     first_branch,
                                     am_branch,
                                     connection_block,
                                     merged_branch,
                                     fc,
                                     sigmoid_out_flag)
    return sam_model, puller


class AttentionModuleModel(nn.Module):

    def __init__(self,
                 basis_branch: nn.Module,
                 first_branch: nn.Module,
                 am_branch: nn.Module,
                 connection_block: nn.Module,
                 merged_branch: nn.Module,
                 fc: nn.Module,
                 sigmoid_out_flag: bool):
        super(AttentionModuleModel, self).__init__()
        self.basis_branch = basis_branch
        # parallel
        self.first_branch = first_branch
        self.am_branch = am_branch
        self.cb = connection_block
        self.merged_branch = merged_branch
        self.fc = fc
        self.is_loss_sigmoid = sigmoid_out_flag
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_images: torch.Tensor) -> tuple:
        basis_out = self.basis_branch(input_images)

        first_out = self.first_branch(basis_out)
        am_out = self.am_branch(basis_out)
        if self.is_loss_sigmoid:
            am_out_sigmoid = self.sigmoid(am_out)
        else:
            am_out_sigmoid = am_out

        connection_out = self.cb(am_out, first_out)

        merged_out = self.merged_branch(connection_out)
        flatten_out = torch.flatten(merged_out, 1)
        classifier_result = self.fc(flatten_out)
        classifier_result_sigmoid = self.sigmoid(classifier_result)
        return classifier_result_sigmoid, am_out_sigmoid


if __name__ == "__main__":
    model, puller = build_attention_module_model(5, m.resnet34(pretrained=True), cb.ConnectionSumBlock())
    image = torch.ones((4, 3, 224, 224))
    segments = torch.ones((4, 5, 224, 224))
    print(model)
    loss = nn.BCELoss()
    c, s = model(image)
    print(loss(s, puller(segments)))
