import torch
import torch.nn as nn
import torchvision.models as m

"""
Давайте разобьем модель на 4 + части
1 -- та что до бранчевания
2 -- левай ветка
3 -- правая ветка
4 -- их соедниение + все остальное

давайте передавать сети типа vgg*

в любом случае я хочу умножать ветки, даже если сегментов нет
"""


def build_attention_module_model(classes: int, pretrained=True):
    """
    build model MAIN FUNCTION HERE!!!!
    :param classes:  class count
    :return: builded model
    """
    sam_branch = nn.Sequential(
        *m.vgg16(pretrained=pretrained).features[2:15],
        nn.Conv2d(256, classes, kernel_size=(3, 3), padding=(1, 1))
        # ,
        # nn.Sigmoid()
    )
    model = m.vgg16(pretrained=True)
    basis_branch = model.features[:4]

    # parallel
    classifier_branch = model.features[4:16]

    merged_branch = model.features[16:]
    merged_branch = nn.Sequential(
        nn.Conv2d(256 * classes, 256, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
        *merged_branch
    )

    avg_pool = model.avgpool
    classifier = nn.Sequential(*model.classifier,
                               nn.Linear(1000, classes),
                               nn.Sigmoid())

    sam_model = AttentionModuleModel(basis_branch,
                                     sam_branch,
                                     classifier_branch,
                                     merged_branch,
                                     avg_pool,
                                     classifier)

    return sam_model


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
                 basis_branch: nn.Module,
                 sam_branch: nn.Module,
                 classification_branch: nn.Module,
                 merged_branch: nn.Module,
                 avg_pool: nn.Module,
                 classification_pool: nn.Module
                 ):
        super(AttentionModuleModel, self).__init__()
        self.basis = basis_branch
        # parallel
        self.classifier_branch = classification_branch
        self.sam_branch = sam_branch

        self.merged_branch = merged_branch
        self.avg_pool = avg_pool
        self.classifier = classification_pool

    def forward(self, input_images: torch.Tensor, segments=None) -> tuple:
        basis_out = self.basis(input_images)
        classifier_out = self.classifier_branch(basis_out)
        sam_out = self.sam_branch(basis_out)
        # замечание -- sam_out -- небольшого размера...
        point_wise_out = None

        for c in range(0, sam_out.size(1)):
            sub_sam_out = sam_out[:, c:c + 1, :, :]
            if point_wise_out is None:
                point_wise_out = classifier_out + sub_sam_out
            else:
                point_wise_out = torch.cat((point_wise_out, classifier_out + sub_sam_out), dim=1)

        merged_out = self.merged_branch(point_wise_out)
        avg_out = self.avg_pool(merged_out)
        flatten_out = torch.flatten(avg_out, 1)
        classifier_result = self.classifier(flatten_out)
        return classifier_result, sam_out
