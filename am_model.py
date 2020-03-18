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


def build_attention_module_model(classes: int):
    """
    build model MAIN FUNCTION HERE!!!!
    :param classes:  class count
    :return: builded model
    """
    sam_branch = nn.Sequential(
        *m.vgg16(pretrained=True).features[2:15],
        nn.Conv2d(256, classes, kernel_size=(3, 3), padding=(1, 1)),
        nn.Sigmoid()
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
        #
        # Попробую схитрить: буду умножать не на то что получилось, а на настоящую сегментационную карту
        #
        basis_out = self.basis(input_images)
        classifier_out = self.classifier_branch(basis_out)
        sam_out = self.sam_branch(basis_out)
        # замечание -- sam_out -- небольшого размера...
        point_wise_out = None
        if segments is None:
            for c in range(0, sam_out.size(1)):
                sub_sam_out = sam_out[:, c:c + 1, :, :]
                if point_wise_out is None:
                    point_wise_out = classifier_out * sub_sam_out
                else:
                    point_wise_out = torch.cat((point_wise_out, classifier_out * sub_sam_out), dim=1)
        else:
            for c in range(0, segments.size(1)):
                sub_sam_out = segments[:, c:c + 1, :, :]
                if point_wise_out is None:
                    point_wise_out = classifier_out * sub_sam_out
                else:
                    point_wise_out = torch.cat((point_wise_out, classifier_out * sub_sam_out), dim=1)
        # if sam_out.shape != classifier_out.shape:
        #    raise Exception("sam output: {} must be equal with classifier output: {}".format(sam_out.shape,
        #                                                                                     classifier_out.shape))
        # point_wise_out = classifier_out * sam_out

        merged_out = self.merged_branch(point_wise_out)
        avg_out = self.avg_pool(merged_out)
        flatten_out = torch.flatten(avg_out, 1)
        classifier_result = self.classifier(flatten_out)
        return classifier_result, sam_out


"""
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    here!
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    here!
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    here!
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    here!
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    here!
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avg_pool): Adaptiveavg_pool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
"""

"""

vgg19

VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace=True)
    (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (24): ReLU(inplace=True)
    (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (26): ReLU(inplace=True)
    (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): ReLU(inplace=True)
    (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (33): ReLU(inplace=True)
    (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (35): ReLU(inplace=True)
    (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avg_pool): Adaptiveavg_pool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
"""
