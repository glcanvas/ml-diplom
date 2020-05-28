import torch
import torch.nn as nn
import torchvision.models as m

import sys

sys.path.insert(0, "/home/nduginec/ml3/ml-diplom")
sys.path.insert(0, "/home/ubuntu/ml3/ml-diplom")

import model.connection_block as cb

"""
Давайте разобьем модель на 4 + части
1 -- та что до бранчевания
2 -- левай ветка
3 -- правая ветка
4 -- их соедниение + все остальное

давайте передавать сети типа vgg*

в любом случае я хочу умножать ветки, даже если сегментов нет
"""


def build_attention_module_model(classes: int, connection_block: nn.Module, pretrained=True):
    """
    build model MAIN FUNCTION HERE!!!!
    :param classes:  class count
    :return: builded model
    """

    if isinstance(connection_block, cb.ConnectionProductBlock):
        sam_branch = nn.Sequential(
            *m.vgg16(pretrained=pretrained).features[:15],
            nn.Conv2d(256, classes, kernel_size=(3, 3), padding=(1, 1)),
            nn.Sigmoid()
        )
        sigmoid_out_flag = False
    else:
        sam_branch = nn.Sequential(
            *m.vgg16(pretrained=pretrained).features[0:15],
            nn.Conv2d(256, classes, kernel_size=(3, 3), padding=(1, 1))
        )
        sigmoid_out_flag = True

    model = m.vgg16(pretrained=True)

    # parallel
    classifier_branch = model.features[0:16]

    merged_branch = model.features[16:]
    merged_branch = nn.Sequential(
        nn.Conv2d(256 * classes, 256, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
        *merged_branch
    )

    avg_pool = model.avgpool
    classifier = nn.Sequential(*model.classifier,
                               nn.Linear(1000, classes),
                               nn.Sigmoid())

    sam_model = AttentionModuleModel(sam_branch,
                                     classifier_branch,
                                     merged_branch,
                                     connection_block,
                                     avg_pool,
                                     classifier, sigmoid_out_flag)
    puller = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), nn.MaxPool2d(kernel_size=2, stride=2))
    return sam_model, puller


class AttentionModuleModel(nn.Module):
    """
    basis_branch: nn.Module
    sam_branch: nn.Module
    classification_branch: nn.Module
    merged_branch: nn.Module
    avg_pool: nn.Module
    classification_pool: nn.Module
    """

    def __init__(self,
                 sam_branch: nn.Module,
                 classification_branch: nn.Module,
                 merged_branch: nn.Module,
                 connection_block: nn.Module,
                 avg_pool: nn.Module,
                 classification_pool: nn.Module,
                 is_loss_sigmoid: bool
                 ):
        super(AttentionModuleModel, self).__init__()
        # parallel
        self.classifier_branch = classification_branch
        self.sam_branch = sam_branch
        self.connection_block = connection_block

        self.merged_branch = merged_branch
        self.avg_pool = avg_pool
        self.classifier = classification_pool

        self.sigmoid = nn.Sigmoid()
        self.is_loss_sigmoid = is_loss_sigmoid

    def forward(self, input_images: torch.Tensor) -> tuple:
        classifier_out = self.classifier_branch(input_images)
        sam_out = self.sam_branch(input_images)

        if self.is_loss_sigmoid:
            sam_out_sigmoid = self.sigmoid(sam_out)
        else:
            sam_out_sigmoid = sam_out

        point_wise_out = self.connection_block(sam_out, classifier_out)
        merged_out = self.merged_branch(point_wise_out)
        avg_out = self.avg_pool(merged_out)
        flatten_out = torch.flatten(avg_out, 1)
        classifier_result = self.classifier(flatten_out)
        return classifier_result, sam_out_sigmoid


if __name__ == "__main__":
    model, puller = build_attention_module_model(5, cb.ConnectionSumBlock())
    image = torch.ones((4, 3, 224, 224))
    segments = torch.ones((4, 5, 224, 224))
    print(model)
    loss = nn.BCELoss()
    c, s = model(image)
    print(loss(s, puller(segments)))
